"""픽셀 캐시를 읽는 Dataset.

캐시는 샤드 9개로 나뉜 (500, 8, 9, 224, 224) uint8 배열이다.
mmap 으로 열어 필요한 스터디만 꺼낸다 (검증: 최대 RSS 0.16 GiB).

읽기 속도가 병목이다. 마운트 검증에서 랜덤 접근이 78.4 ms/study 로 측정됐다
(1 epoch 읽기만 5.8분). 두 가지로 줄인다

- DataLoader num_workers 병렬 읽기
- 샤드 단위 순차 접근 (ShardShuffleSampler) - 샤드 안에서만 섞는다
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

import config
import dicom_io


@dataclass
class CacheHandle:
    root: Path
    meta: dict
    index: pd.DataFrame          # study, shard, row, 슬롯 채움 플래그
    arrays: list                 # mmap 배열 9개

    @property
    def slots(self) -> list[str]:
        return list(self.meta["slots"])

    def get(self, shard: int, row: int) -> np.ndarray:
        return np.asarray(self.arrays[shard][row])      # (8, 9, 224, 224)


def open_cache(root: Path | str) -> CacheHandle:
    root = Path(root)
    meta = json.loads((root / "cache_meta.json").read_text())
    index = pd.read_csv(root / "cache_index.csv")
    shards = sorted(root.glob("cache_*.npy"))
    if len(shards) != meta["n_shards"]:
        raise RuntimeError(f"샤드 {len(shards)}개, 메타는 {meta['n_shards']}개")
    arrays = [np.load(f, mmap_mode="r") for f in shards]
    if meta["slots"] != list(dicom_io.SLOTS):
        raise RuntimeError(
            f"캐시 슬롯이 현재 코드와 다르다\n  캐시: {meta['slots']}\n  코드: {list(dicom_io.SLOTS)}")
    return CacheHandle(root=root, meta=meta, index=index, arrays=arrays)


# ---------------------------------------------------------------------------
# 슬롯 구성 - A안(평면 고정) / C안(평면 통합) 을 캐시 재빌드 없이 고른다
# ---------------------------------------------------------------------------
PLAN_A = ("SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
          "SAG_FLUID_NOFS", "COR_T1", "SAG_T1")

# C안: 평면을 통합해 빈칸을 줄인다
#   T1 보유 스터디는 92.6% 인데 시상면 T1 로 좁히면 40.4% 다 (docs/eda.md)
PLAN_C = (
    ("SAG_FLUID_FS",),
    ("COR_FLUID_FS",),
    ("AX_FLUID_FS",),
    ("SAG_FLUID_NOFS", "FLUID_NOFS_EXTRA"),          # 있는 것을 쓴다
    ("COR_T1", "SAG_T1", "T1_EXTRA"),                # T1_1
    ("SAG_T1", "T1_EXTRA", "COR_T1"),                # T1_2 (T1_1 과 다른 것)
)


def resolve_plan(plan: str) -> tuple[tuple[str, ...], ...]:
    if plan == "A":
        return tuple((s,) for s in PLAN_A)
    if plan == "C":
        return PLAN_C
    if plan == "ALL8":
        return tuple((s,) for s in dicom_io.SLOTS)
    raise ValueError(f"알 수 없는 plan: {plan}")


def build_slot_map(cache: CacheHandle, plan: str) -> tuple[np.ndarray, np.ndarray]:
    """스터디마다 각 출력 칸에 캐시의 몇 번 슬롯을 넣을지 정한다.

    돌려주는 것
      pick  (N, n_out) int8   캐시 슬롯 인덱스, 없으면 -1
      mask  (N, n_out) float32  채워졌으면 1
    """
    prefs = resolve_plan(plan)
    slots = cache.slots
    filled = cache.index[slots].values.astype(bool)       # (N, 8)
    n, n_out = len(filled), len(prefs)

    pick = np.full((n, n_out), -1, dtype=np.int8)
    for j, pref in enumerate(prefs):
        used = pick[:, :j]
        for name in pref:
            k = slots.index(name)
            free = (pick[:, j] < 0) & filled[:, k]
            if j > 0:                                     # 앞 칸이 이미 쓴 슬롯은 피한다
                free &= ~(used == k).any(axis=1)
            pick[free, j] = k
    mask = (pick >= 0).astype(np.float32)
    return pick, mask


# ---------------------------------------------------------------------------
class KneeCacheDataset(Dataset):
    """스터디 하나 -> (n_out, 9, 224, 224) float32, 타깃, 가중, 슬롯 마스크."""

    def __init__(self, cache: CacheHandle, studies: list[str],
                 targets: pd.DataFrame | None, weights: pd.DataFrame | None,
                 plan: str = "C", augment: bool = False, seed: int = 0):
        self.cache = cache
        self.plan = plan
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        idx = cache.index.set_index("study")
        missing = [s for s in studies if s not in idx.index]
        if missing:
            raise KeyError(f"캐시에 없는 스터디 {len(missing)}건 (예: {missing[0][-12:]})")

        self.studies = list(studies)
        self.loc = idx.loc[self.studies, ["shard", "row"]].values.astype(int)

        pick, mask = build_slot_map(cache, plan)
        pos = {s: i for i, s in enumerate(cache.index.study)}
        rows = [pos[s] for s in self.studies]
        self.pick = pick[rows]
        self.mask = mask[rows]

        L = list(config.LABELS)
        self.y = (targets.loc[self.studies, L].values.astype(np.float32)
                  if targets is not None else None)
        self.w = (weights.loc[self.studies, L].values.astype(np.float32)
                  if weights is not None else None)

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, i: int):
        shard, row = self.loc[i]
        full = self.cache.get(shard, row)             # (8, 9, 224, 224) uint8
        pick = self.pick[i]

        out = np.zeros((len(pick), full.shape[1], full.shape[2], full.shape[3]),
                       dtype=np.uint8)
        for j, k in enumerate(pick):
            if k >= 0:
                out[j] = full[k]

        if self.augment:
            out = self._augment(out)

        # uint8 로 넘기고 정규화는 GPU 에서 한다 (디스커션 743374)
        item = {
            "image": torch.from_numpy(np.ascontiguousarray(out)),
            "slot_mask": torch.from_numpy(self.mask[i]),
            "study": self.studies[i],
        }
        if self.y is not None:
            item["y"] = torch.from_numpy(self.y[i])
            item["w"] = torch.from_numpy(self.w[i])
        return item

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """가벼운 증강만 쓴다.

        좌우 뒤집기는 넣지 않는다. medial / lateral 이 좌우로 정의된 타깃이라
        뒤집으면 레이블이 바뀐다.
        """
        if self.rng.random() < 0.5:                        # 상하 뒤집기도 해부학을 깨므로 제외
            pass
        if self.rng.random() < 0.5:                        # 밝기/대비
            g = self.rng.uniform(0.85, 1.15)
            x = np.clip(x.astype(np.float32) * g, 0, 255).astype(np.uint8)
        if self.rng.random() < 0.3:                        # 슬라이스 방향 뒤집기 (순서만)
            x = x[:, ::-1]
        return np.ascontiguousarray(x)


class ShardSequentialSampler(Sampler):
    """샤드 단위로 순차 접근해 랜덤 읽기 비용을 줄인다.

    샤드 순서와 샤드 내부 순서는 매 epoch 섞는다.
    완전 무작위보다 지역성이 좋아 mmap 읽기가 빨라진다.
    """

    def __init__(self, dataset: KneeCacheDataset, shuffle: bool = True, seed: int = 0):
        self.n = len(dataset)
        self.shards = dataset.loc[:, 0]
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = []
        uniq = np.unique(self.shards)
        if self.shuffle:
            uniq = rng.permutation(uniq)
        for s in uniq:
            idx = np.flatnonzero(self.shards == s)
            if self.shuffle:
                idx = rng.permutation(idx)
            order.append(idx)
        return iter(np.concatenate(order).tolist())
