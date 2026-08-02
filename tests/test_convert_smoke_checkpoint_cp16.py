from __future__ import annotations

from collections import OrderedDict
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "convert_smoke_checkpoint_cp16",
    ROOT / "tools" / "convert_smoke_checkpoint_cp16.py",
)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _states():
    generator = torch.Generator().manual_seed(19)
    native = torch.randn(3, 2, generator=generator)
    appearance = torch.randn(16, 256, generator=generator)
    source = OrderedDict(
        (
            ("head.native.weight", native),
            (converter.APPEARANCE_KEY, appearance),
            (converter.OUTPUT_KEY, torch.zeros(256, 144)),
        )
    )
    source._metadata = OrderedDict((("", {"version": 1}),))
    target = OrderedDict(
        (
            ("head.native.weight", torch.empty_like(native)),
            (converter.APPEARANCE_KEY, torch.empty_like(appearance)),
            (converter.RAY_KEY, torch.empty(16, 9)),
            (converter.OUTPUT_KEY, torch.empty(256, 16)),
        )
    )
    target._metadata = OrderedDict(
        (("", {"version": 2}), ("head.rosa_refiner.rosa.rfl", {"version": 1}))
    )
    return source, target


def test_zero_dense_rfl_conversion_is_deterministic_and_schema_exact():
    source, target = _states()
    first = converter.convert_state_dict(source, target)
    second = converter.convert_state_dict(source, target)

    assert list(first) == list(target)
    assert torch.equal(first["head.native.weight"], source["head.native.weight"])
    assert torch.equal(
        first[converter.APPEARANCE_KEY], source[converter.APPEARANCE_KEY]
    )
    assert torch.count_nonzero(first[converter.OUTPUT_KEY]).item() == 0
    assert torch.count_nonzero(first[converter.RAY_KEY]).item() > 0
    assert torch.equal(first[converter.RAY_KEY], second[converter.RAY_KEY])
    assert first._metadata == target._metadata
    assert tuple(source[converter.OUTPUT_KEY].shape) == (256, 144)


def test_nonzero_dense_output_is_never_claimed_convertible():
    source, target = _states()
    source[converter.OUTPUT_KEY][0, 0] = 1.0
    with pytest.raises(converter.ConversionError, match="non-zero"):
        converter.convert_state_dict(source, target)


@pytest.mark.parametrize("failure", ("source_extra", "target_extra", "shape"))
def test_only_the_two_declared_rfl_topology_changes_are_allowed(failure):
    source, target = _states()
    if failure == "source_extra":
        source["unexpected.weight"] = torch.ones(1)
        match = "outside the target schema"
    elif failure == "target_extra":
        target["new.weight"] = torch.ones(1)
        match = "more than the CP16 ray projection"
    else:
        source["head.native.weight"] = torch.ones(4, 2)
        match = "unchanged state"
    with pytest.raises(converter.ConversionError, match=match):
        converter.convert_state_dict(source, target)


def test_atomic_save_is_no_clobber_by_default(tmp_path: Path):
    destination = tmp_path / "fixture.pth"
    payload = {"meta": {"benchmark_only": True}, "state_dict": {"x": torch.ones(1)}}
    written = converter._atomic_save(payload, destination, overwrite=False)
    assert written == destination
    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    assert loaded["meta"] == {"benchmark_only": True}
    assert torch.equal(loaded["state_dict"]["x"], torch.ones(1))
    with pytest.raises(FileExistsError):
        converter._atomic_save(payload, destination, overwrite=False)


def test_source_contract_requires_training_payload_and_balanced_v2_metadata():
    from dataclasses import asdict
    from rosa.config import ROSAConfig

    source, _ = _states()
    checkpoint = {
        "state_dict": source,
        "meta": {
            "rosa_contract_version": 2,
            "refinement_profile": "balanced",
            "rfl_topology_contract": None,
            "projection_contract": "full_fov",
            "rosa_config": asdict(ROSAConfig.from_mapping({"profile": "balanced"})),
        },
        "optimizer": {},
        "param_schedulers": [],
        "message_hub": {},
    }
    state, metadata = converter._validate_source_checkpoint(checkpoint)
    assert state is source
    assert metadata is checkpoint["meta"]
    checkpoint.pop("optimizer")
    with pytest.raises(converter.ConversionError, match="training payload"):
        converter._validate_source_checkpoint(checkpoint)


def test_expected_hash_is_strict_lowercase_hex(tmp_path: Path):
    source = tmp_path / "source.pth"
    source.write_bytes(b"not reached")
    with pytest.raises(ValueError, match="64 lowercase hex"):
        converter.convert_checkpoint(
            source,
            tmp_path / "output.pth",
            expected_source_sha256="A" * 64,
        )
