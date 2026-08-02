from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from dataclasses import asdict
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "analyze_stage_routes",
    Path(__file__).parents[1] / "tools" / "analyze_stage_routes.py",
)
assert _SPEC is not None and _SPEC.loader is not None
stage_routes = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stage_routes
_SPEC.loader.exec_module(stage_routes)


def _head(**overrides):
    values = {
        "training": False,
        "num_decoder": 6,
        "max_temporal_gnn_interactions": 0,
        "instance_bank": SimpleNamespace(num_temp_instances=0),
        "sampler": SimpleNamespace(num_temp_dn_groups=0, dn_metas=None),
        "_diagnostic_return_all_cls": True,
        "operation_order": ["deformable", "ffn", "norm", "refine"]
        + ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _native_batch(batch=2, queries=3):
    predictions = []
    captures = []
    current_anchor = (
        torch.arange(queries, dtype=torch.float32)[None, :, None]
        .expand(batch, -1, 11)
        .clone()
    )
    for stage in range(6):
        captures.append(
            stage_routes.CapturedRefineInput(
                query=torch.full((batch, queries, 256), float(stage)),
                anchor=current_anchor,
            )
        )
        current_anchor = current_anchor + float(stage + 1)
        predictions.append(current_anchor)
    outputs = {
        "prediction": predictions,
        "classification": [
            torch.full((batch, queries, 6), float(stage)) for stage in range(6)
        ],
        "quality": [
            torch.full((batch, queries, 2), float(stage + 10)) for stage in range(6)
        ],
        "instance_id": torch.arange(batch * queries).reshape(batch, queries),
    }
    return outputs, captures


def _sample(token="sample-0"):
    prediction = torch.stack([torch.full((3, 11), float(stage)) for stage in range(6)])
    classification = torch.stack(
        [torch.full((3, 2), float(10 + stage)) for stage in range(6)]
    )
    quality = torch.stack([torch.full((3, 2), float(20 + stage)) for stage in range(6)])
    return {
        "token": token,
        "sample_idx": token,
        "prediction": prediction,
        "classification": classification,
        "quality": quality,
        "instance_id": torch.arange(3),
        "ground_truth": {
            "boxes": torch.zeros(1, 9),
            "labels": torch.zeros(1, dtype=torch.long),
            "box_origin": [0.5, 0.5, 0.0],
        },
    }


class _Decoder:
    num_output = 3
    score_threshold = None
    sorted = True
    selection_score_mode = "legacy"
    duplicate_bev_radius = 0.0

    def __init__(self):
        self.calls = []

    def decode(
        self,
        classifications,
        predictions,
        instance_id,
        qualities,
        output_idx,
    ):
        self.calls.append(
            {
                "box": float(predictions[output_idx][0, 0, 0]),
                "classification": float(classifications[output_idx][0, 0, 0]),
                "quality": float(qualities[output_idx][0, 0, 0]),
                "instance_id": instance_id.clone(),
            }
        )
        return [
            {
                "boxes_3d": predictions[output_idx][0, :1, :9],
                "scores_3d": torch.ones(1),
                "labels_3d": torch.zeros(1, dtype=torch.long),
                "query_indices": torch.zeros(1, dtype=torch.long),
                # This is intentionally wrong and must never leave the router.
                "cls_scores": torch.full((1,), -999.0),
            }
        ]


def _payload(samples=None):
    decoder = _Decoder()
    return stage_routes.make_dump_payload(
        samples=samples or [_sample()],
        dataset_meta={
            "classes": ["Car"],
            "version": "test",
            "categories": {"Car": 0},
        },
        operation_order=_head().operation_order,
        decoder=decoder,
        config_path=Path("config.py"),
        config_fingerprint="a" * 64,
        checkpoint_path=Path("model.pth"),
        checkpoint_sha256="b" * 64,
    )


def test_route_matrices_are_complete_unique_and_global():
    score = stage_routes.route_specs(stage_routes.MATRIX_SCORE_BUNDLE)
    aligned = stage_routes.route_specs(stage_routes.MATRIX_BOX_ALIGNED_QUALITY)
    assert len(score) == len(aligned) == 36
    assert len({route.key for route in score}) == 36
    assert len({route.key for route in aligned}) == 36
    assert all(route.classification_stage == route.quality_stage for route in score)
    assert all(route.box_stage == route.quality_stage for route in aligned)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"training": True}, "eval"),
        (
            {"instance_bank": SimpleNamespace(num_temp_instances=1)},
            "num_temp_instances",
        ),
        (
            {"sampler": SimpleNamespace(num_temp_dn_groups=1, dn_metas=None)},
            "num_temp_dn_groups",
        ),
        (
            {"sampler": SimpleNamespace(num_temp_dn_groups=0, dn_metas={})},
            "denoising state",
        ),
        ({"_diagnostic_return_all_cls": False}, "diagnostic"),
    ],
)
def test_head_contract_fails_closed_for_unsafe_runtime(override, message):
    with pytest.raises(RuntimeError, match=message):
        stage_routes.validate_head_contract(_head(**override))


def test_head_contract_accepts_exact_temporal_and_dn_free_eval():
    indices = stage_routes.validate_head_contract(_head())
    assert indices == (3, 9, 15, 21, 27, 33)


def test_native_batch_proves_row_identity_and_all_scores_materialize():
    outputs, captures = _native_batch()
    native = stage_routes.validate_native_batch(outputs, captures)
    assert set(native) == {"prediction", "classification", "quality"}
    assert all(len(values) == 6 for values in native.values())


def test_native_batch_rejects_row_reordering_and_dn_leak():
    outputs, captures = _native_batch()
    bad_anchor = captures[2].anchor.flip(1)
    captures[2] = stage_routes.CapturedRefineInput(
        captures[2].query,
        bad_anchor,
    )
    with pytest.raises(RuntimeError, match="row identity failed"):
        stage_routes.validate_native_batch(outputs, captures)

    outputs, captures = _native_batch()
    outputs["dn_prediction"] = []
    with pytest.raises(RuntimeError, match="DN state"):
        stage_routes.validate_native_batch(outputs, captures)


def test_native_batch_rejects_missing_intermediate_classification():
    outputs, captures = _native_batch()
    outputs["classification"][3] = None
    with pytest.raises(RuntimeError, match="diagnostic all-cls"):
        stage_routes.validate_native_batch(outputs, captures)


def test_pack_raw_batch_stores_d0_d5_once_per_sample_on_cpu():
    outputs, captures = _native_batch(batch=2, queries=3)
    data_samples = [
        SimpleNamespace(
            metainfo={"token": f"token-{index}", "sample_idx": index},
            eval_ann_info={
                "gt_bboxes_3d": SimpleNamespace(tensor=torch.zeros(index, 9)),
                "gt_labels_3d": torch.zeros(index, dtype=torch.long),
            },
        )
        for index in range(2)
    ]
    packed = stage_routes.pack_raw_batch(outputs, data_samples, captures)
    assert [sample["token"] for sample in packed] == ["token-0", "token-1"]
    assert all(sample["prediction"].shape == (6, 3, 11) for sample in packed)
    assert all(sample["classification"].shape == (6, 3, 6) for sample in packed)
    assert all(sample["quality"].shape == (6, 3, 2) for sample in packed)
    assert all(sample["prediction"].device.type == "cpu" for sample in packed)
    assert all(
        sample["ground_truth"]["box_origin"] == [0.5, 0.5, 0.0] for sample in packed
    )


def test_dump_schema_is_weights_only_loadable_and_detects_tampering(tmp_path):
    payload = _payload()
    path = tmp_path / "routes.pt"
    torch.save(payload, path)
    loaded = stage_routes.load_dump(path)
    assert loaded["schema"] == stage_routes.DUMP_SCHEMA
    assert loaded["contract"]["emitted_cls_scores_stored"] is False

    loaded["contract"]["row_identity_verified"] = False
    with pytest.raises(ValueError, match="row_identity_verified"):
        stage_routes.validate_dump_payload(loaded)


def test_ground_truth_canonical_box_round_trip_preserves_gravity_center():
    boxes_module = pytest.importorskip("mmdet3d.structures")
    box_type = boxes_module.LiDARInstance3DBoxes
    source = box_type(
        torch.tensor([[1.0, 2.0, 3.0, 2.0, 4.0, 1.6, 0.2, 0.0, 0.0]]),
        box_dim=9,
        origin=(0.5, 0.5, 0.5),
    )
    data_sample = SimpleNamespace(
        eval_ann_info={
            "gt_bboxes_3d": source,
            "gt_labels_3d": torch.tensor([0]),
        }
    )
    packed = stage_routes._sample_ground_truth(data_sample)
    rebuilt = box_type(
        packed["boxes"],
        box_dim=packed["boxes"].shape[-1],
        origin=tuple(packed["box_origin"]),
    )
    torch.testing.assert_close(rebuilt.tensor, source.tensor)
    torch.testing.assert_close(rebuilt.gravity_center, source.gravity_center)


def test_raw_route_uses_requested_sources_and_drops_emitted_cls_scores():
    decoder = _Decoder()
    route = stage_routes.RouteSpec(
        stage_routes.MATRIX_SCORE_BUNDLE,
        box_stage=4,
        classification_stage=2,
        quality_stage=3,
    )
    decoded = stage_routes.decode_raw_route(decoder, _sample(), route)
    assert len(decoder.calls) == 1
    call = decoder.calls[0]
    assert call["box"] == 4.0
    assert call["classification"] == 12.0
    assert call["quality"] == 23.0
    torch.testing.assert_close(call["instance_id"], torch.arange(3).unsqueeze(0))
    assert set(decoded) == {"boxes_3d", "scores_3d", "labels_3d"}
    assert "cls_scores" not in decoded


def test_metric_records_apply_one_route_to_every_sample():
    samples = [_sample("a"), _sample("b")]
    payload = _payload(samples)
    decoder = _Decoder()
    route = stage_routes.RouteSpec(
        stage_routes.MATRIX_BOX_ALIGNED_QUALITY,
        box_stage=3,
        classification_stage=5,
        quality_stage=3,
    )
    records = stage_routes.routed_metric_records(
        payload,
        decoder,
        route,
        box_factory=lambda tensor, _origin: tensor,
    )
    assert [record["token"] for record in records] == ["a", "b"]
    assert len(decoder.calls) == 2
    assert all(
        call["box"] == 3.0
        and call["classification"] == 15.0
        and call["quality"] == 23.0
        for call in decoder.calls
    )
    assert all(
        set(record["pred_instances_3d"]) == {"bboxes_3d", "scores_3d", "labels_3d"}
        for record in records
    )


def test_cli_exposes_dump_and_evaluate_contracts(tmp_path):
    dump = stage_routes.parse_args(
        [
            "dump",
            "config.py",
            "model.pth",
            "--output",
            str(tmp_path / "raw.pt"),
            "--max-samples",
            "4",
        ]
    )
    assert dump.command == "dump"
    assert dump.max_samples == 4

    evaluate = stage_routes.parse_args(
        [
            "evaluate",
            "config.py",
            str(tmp_path / "raw.pt"),
            "--output",
            str(tmp_path / "metrics.json"),
            "--matrix",
            stage_routes.MATRIX_SCORE_BUNDLE,
        ]
    )
    assert evaluate.command == "evaluate"
    assert evaluate.matrix == stage_routes.MATRIX_SCORE_BUNDLE


def test_dump_rejects_partial_factorized_checkpoint_before_runner_build(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("mmengine")
    from mmengine.config import Config
    from rosa import ROSAConfig

    effective = ROSAConfig.from_mapping({"profile": "balanced"})
    checkpoint = tmp_path / "partial.pth"
    torch.save(
        {
            "state_dict": {
                "head.rosa_refiner.rosa.rfl.output_proj.weight": torch.ones(1)
            },
            "meta": {
                "rosa_contract_version": 3,
                "rfl_topology_contract": effective.rfl_topology_contract,
                "projection_contract": effective.projection_contract,
                "rosa_config": asdict(effective),
            },
        },
        checkpoint,
    )
    config = Config(
        {
            "model": {
                "head": {
                    "type": "ROSASparse4DHead",
                    "rosa_cfg": {"profile": "balanced"},
                }
            }
        }
    )
    monkeypatch.setattr(stage_routes, "_config_from_args", lambda _args: config)
    checkpoint_tool = stage_routes._checkpoint_run_tool()
    snapshot_factory = checkpoint_tool._private_checkpoint_snapshot
    observed = {}

    @contextmanager
    def observe_snapshot(path):
        with snapshot_factory(path) as snapshot:
            observed["snapshot"] = snapshot
            yield snapshot

    monkeypatch.setattr(
        checkpoint_tool,
        "_private_checkpoint_snapshot",
        observe_snapshot,
    )

    from mmengine.runner import Runner

    def must_not_build(*_args, **_kwargs):
        raise AssertionError("invalid checkpoint reached Runner construction")

    monkeypatch.setattr(Runner, "from_cfg", must_not_build)
    args = SimpleNamespace(
        config=tmp_path / "config.py",
        checkpoint=checkpoint,
        output=tmp_path / "routes.pt",
        work_dir=None,
        cfg_options=None,
    )

    with pytest.raises(ValueError, match="topology integrity.*missing keys"):
        stage_routes.dump_command(args)
    snapshot = observed["snapshot"]
    assert not snapshot.path.exists()
    assert not snapshot.private_directory.exists()


def _save_factorized_checkpoint(path: Path) -> None:
    from rosa import ROSAConfig, ROSARefiner

    effective = ROSAConfig.from_mapping({"profile": "balanced"})
    with torch.random.fork_rng(devices=[]):
        state = ROSARefiner(effective).state_dict()
    torch.save(
        {
            "state_dict": {
                "head.rosa_refiner." + key: value for key, value in state.items()
            },
            "meta": {
                "rosa_contract_version": 3,
                "rfl_topology_contract": effective.rfl_topology_contract,
                "projection_contract": effective.projection_contract,
                "rosa_config": asdict(effective),
            },
        },
        path,
    )


@pytest.mark.parametrize("attack", ("rename_replace", "in_place_overwrite"))
def test_dump_loads_verified_snapshot_after_source_path_attack_and_cleans_up(
    tmp_path,
    monkeypatch,
    attack,
):
    pytest.importorskip("mmengine")
    from mmengine.config import Config
    from mmengine.runner import Runner

    checkpoint = tmp_path / "valid.pth"
    _save_factorized_checkpoint(checkpoint)
    original_inode = checkpoint.stat().st_ino
    config = Config(
        {
            "model": {
                "head": {
                    "type": "ROSASparse4DHead",
                    "rosa_cfg": {"profile": "balanced"},
                }
            }
        }
    )
    monkeypatch.setattr(stage_routes, "_config_from_args", lambda _args: config)

    checkpoint_tool = stage_routes._checkpoint_run_tool()
    snapshot_factory = checkpoint_tool._private_checkpoint_snapshot
    validate = stage_routes._validate_checkpoint_contract
    observed = {}

    @contextmanager
    def observe_snapshot(path):
        with snapshot_factory(path) as snapshot:
            observed["snapshot"] = snapshot
            yield snapshot

    def validate_then_attack(snapshot_path, received_config):
        validate(snapshot_path, received_config)
        invalid = {
            "state_dict": {"head.decoder.weight": torch.ones(1)},
            "meta": {},
        }
        if attack == "rename_replace":
            replacement = tmp_path / "replacement.pth"
            torch.save(invalid, replacement)
            os.replace(replacement, checkpoint)
            assert checkpoint.stat().st_ino != original_inode
        else:
            torch.save(invalid, checkpoint)
            assert checkpoint.stat().st_ino == original_inode

    class StopAfterLoad(RuntimeError):
        pass

    class FakeRunner:
        def __init__(self, load_from):
            self.load_from = Path(load_from)

        def load_or_resume(self):
            from mmengine.runner.checkpoint import CheckpointLoader

            loaded = CheckpointLoader.load_checkpoint(
                str(self.load_from),
                map_location="cpu",
                logger=None,
            )
            state = checkpoint_tool._checkpoint_state_dict(loaded)
            assert state is not None
            assert (
                len([key for key in state if key.startswith("head.rosa_refiner.")])
                == 40
            )
            assert "head.rosa_refiner.rosa.rfl.ray_proj.weight" in state
            observed["loaded"] = True
            raise StopAfterLoad("stop after sealed analyzer load")

    def build_runner(_runner_cls, received_config):
        snapshot = observed["snapshot"]
        assert received_config.load_from == str(snapshot.path)
        return FakeRunner(received_config.load_from)

    monkeypatch.setattr(
        checkpoint_tool,
        "_private_checkpoint_snapshot",
        observe_snapshot,
    )
    monkeypatch.setattr(
        stage_routes,
        "_validate_checkpoint_contract",
        validate_then_attack,
    )
    monkeypatch.setattr(Runner, "from_cfg", classmethod(build_runner))
    args = SimpleNamespace(
        config=tmp_path / "config.py",
        checkpoint=checkpoint,
        output=tmp_path / "routes.pt",
        work_dir=None,
        cfg_options=None,
    )

    with pytest.raises(StopAfterLoad, match="sealed analyzer load"):
        stage_routes.dump_command(args)

    attacked = checkpoint_tool._load_checkpoint_without_executing_code(checkpoint)
    assert set(attacked["state_dict"]) == {"head.decoder.weight"}
    assert observed["loaded"] is True
    snapshot = observed["snapshot"]
    assert not snapshot.path.exists()
    assert not snapshot.private_directory.exists()
