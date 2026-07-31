from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mmengine")
pytest.importorskip("mmdet3d")
from mmengine.config import Config
from mmengine.dataset import BaseDataset, DefaultSampler
import mmengine.dataset.sampler as sampler_module
import mmengine.runner.utils as runner_utils
from mmengine.runner import set_random_seed
from mmdet3d.datasets import CBGSDataset
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules

from rosa.integrations.fisheye3dod.dataset import (
    Fisheye3DODEvaluatorRangeFilter,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rosa_r18_main126_v2_c0_cbgs_native.py"
EXPECTED_INDEX_HASH = "5d97f92ffe416b91230cfafca20210e63eb7f2ad24c59845c822997ef76e5201"


@pytest.fixture(scope="module")
def c0_config() -> Config:
    register_all_modules(init_default_scope=True)
    return Config.fromfile(CONFIG_PATH)


def _build_seeded_dataset(config: Config) -> CBGSDataset:
    ann_file = Path(config.train_dataloader.dataset.dataset.ann_file)
    if not ann_file.is_file():
        pytest.skip(f"Main126 annotations are unavailable: {ann_file}")
    numpy_state = np.random.get_state()
    try:
        np.random.seed(config.randomness.seed)
        dataset = DATASETS.build(config.train_dataloader.dataset)
    finally:
        np.random.set_state(numpy_state)
    assert isinstance(dataset, CBGSDataset)
    return dataset


@pytest.fixture(scope="module")
def c0_dataset(c0_config: Config) -> CBGSDataset:
    return _build_seeded_dataset(c0_config)


def _sample_indices_hash(dataset: CBGSDataset) -> str:
    indices = np.asarray(dataset.sample_indices, dtype=np.int64)
    return hashlib.sha256(indices.tobytes()).hexdigest()


def test_c0_resolves_to_the_official_raw_cbgs_contract(c0_config: Config) -> None:
    train_dataset = c0_config.train_dataloader.dataset
    contract = c0_config.experiment_contract

    assert train_dataset.type == "CBGSDataset"
    assert train_dataset.dataset.type == "Fisheye3DODSparse4DDataset"
    assert c0_config.model.head.type == "ROSANativeBaselineHead"
    assert c0_config.model.head.rosa_cfg is None
    assert c0_config.train_dataloader.batch_size == 4
    assert c0_config.train_cfg.max_epochs == 20
    assert contract.global_batch_size == 8
    assert contract.raw_train_frames == 8_820
    assert contract.cbgs_train_samples == 25_457
    assert contract.b0_steps_per_epoch == 1_103
    assert contract.cbgs_steps_per_epoch == 3_183
    assert contract.b0_optimizer_steps_20e == 22_060
    assert contract.cbgs_optimizer_steps_20e == 63_660
    assert contract.optimizer_step_increase_20e == 41_600
    assert contract.train_sampling == (
        "mmdet3d.CBGSDataset(raw_pre_pipeline_categories)"
    )


def test_main126_cbgs_length_and_seeded_indices_repeat(
    c0_config: Config,
    c0_dataset: CBGSDataset,
) -> None:
    repeated = _build_seeded_dataset(c0_config)

    assert len(c0_dataset.dataset) == 8_820
    assert len(c0_dataset) == 25_457
    assert c0_dataset.sample_indices == repeated.sample_indices
    assert _sample_indices_hash(c0_dataset) == EXPECTED_INDEX_HASH


def test_get_cat_ids_is_raw_and_precedes_evaluator_range(
    c0_dataset: CBGSDataset,
) -> None:
    native_dataset = c0_dataset.dataset
    # Main126 frame 14 contains a Bus at radius 49.48 m. CBGS sees it, while
    # the strict evaluator-compatible 48 m training range removes it later.
    assert native_dataset.get_cat_ids(14) == {0, 3}

    data_info = native_dataset.get_data_info(14)
    filtered = Fisheye3DODEvaluatorRangeFilter()(data_info)
    assert set(filtered["ann_info"]["gt_labels_3d"].tolist()) == {0}


def test_shared_mmengine_seed_keeps_cbgs_indices_equal_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ToyCategoryDataset(BaseDataset):
        METAINFO = {"classes": ("major", "middle", "rare")}

        def load_data_list(self) -> list[dict]:
            return [
                {"cat_ids": (0,)},
                {"cat_ids": (0,)},
                {"cat_ids": (0, 1)},
                {"cat_ids": (1,)},
                {"cat_ids": (1, 2)},
                {"cat_ids": (2,)},
                {"cat_ids": (2,)},
            ]

        def get_cat_ids(self, idx: int) -> set[int]:
            return set(self.get_data_info(idx)["cat_ids"])

    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    def build_for_rank(rank: int, *, diff_rank_seed: bool) -> list[int]:
        monkeypatch.setattr(runner_utils, "get_rank", lambda: rank)
        set_random_seed(
            3407,
            deterministic=False,
            diff_rank_seed=diff_rank_seed,
        )
        return CBGSDataset(ToyCategoryDataset()).sample_indices

    try:
        rank0 = build_for_rank(0, diff_rank_seed=False)
        rank1 = build_for_rank(1, diff_rank_seed=False)
        assert rank0 == rank1
        assert build_for_rank(0, diff_rank_seed=True) != build_for_rank(
            1,
            diff_rank_seed=True,
        )
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def test_default_sampler_partitions_one_shared_cbgs_index_space(
    c0_dataset: CBGSDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sampler_module, "get_dist_info", lambda: (0, 2))
    rank0 = list(DefaultSampler(c0_dataset, shuffle=True, seed=3407))
    monkeypatch.setattr(sampler_module, "get_dist_info", lambda: (1, 2))
    rank1 = list(DefaultSampler(c0_dataset, shuffle=True, seed=3407))

    generator = torch.Generator().manual_seed(3407)
    shared_order = torch.randperm(len(c0_dataset), generator=generator).tolist()
    shared_order.append(shared_order[0])
    assert rank0 == shared_order[0::2]
    assert rank1 == shared_order[1::2]
