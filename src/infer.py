"""추론과 제출 파일 생성.

테스트 스터디는 캐시가 없다. DICOM 에서 바로 전처리해 모델에 넣는다.

제출 규칙

- 컬럼은 sample_submission.csv 와 **정확히 같은 순서**여야 한다
  어긋나면 AUC 가 0.5 근처로 나오고 에러는 안 난다 (config.make_submission 강제)
- AUC 는 순서만 보므로 확률을 굳이 보정할 필요가 없다
- 여러 모델을 합칠 때는 확률 평균이 아니라 **rank 평균**이다
  확률 평균은 가장 자신만만한 모델이 지배한다
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import cache as cache_mod
import config
import dataset as ds
import dicom_io
import model as M
import paths


@dataclass
class InferConfig:
    plan: str = "C"
    batch_size: int = 8
    amp: bool = True
    tta_slice_flip: bool = False     # 슬라이스 순서 뒤집기 TTA
    n_groups: int | None = None      # None 이면 모델에서 읽는다 (해상도마다 다르다)
    num_workers: int = 2


def build_study_tensor(study_uid: str, plan: str, ccfg: cache_mod.CacheConfig,
                       train: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """DICOM 에서 (n_out, 9, 224, 224) uint8 과 슬롯 마스크를 만든다."""
    arr, info = cache_mod.build_study(study_uid, ccfg, train=train)
    slots = list(ccfg.slots)
    filled = np.array([[s in info.slots_filled for s in slots]])      # (1, n_slots)

    # 배정 규칙은 dataset.assign_slots 하나만 쓴다. 학습과 추론이 갈리면 안 된다
    pick, mask = ds.assign_slots(plan, slots, filled)
    pick, mask = pick[0], mask[0]

    out = np.zeros((len(pick), arr.shape[1], arr.shape[2], arr.shape[3]), np.uint8)
    for j, k in enumerate(pick):
        if k >= 0:
            out[j] = arr[k]
    return out, mask


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[M.KneeModel, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = M.ModelConfig(**ck["model_cfg"])
    mcfg.pretrained = False          # 가중치는 체크포인트에서 온다
    mcfg.weights_path = None
    net = M.KneeModel(mcfg)
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), ck


def config_for(ck: dict) -> cache_mod.CacheConfig:
    """체크포인트가 학습된 전처리 설정을 복원한다.

    **모델마다 전처리가 다르다.** 224 모델은 9슬라이스 224px,
    336 모델은 6슬라이스 336px 로 학습됐다.
    한 설정으로 만든 텐서를 둘 다에 먹이면 조용히 틀린다
    ("체크포인트는 자기 전처리 계약을 갖고 다닌다", 디스커션 735154)
    """
    m = ck["model_cfg"]
    c = cache_mod.CacheConfig(out_px=int(m["img_size"]), n_slices=int(m["n_slices"]))
    c.slots = dicom_io.SLOTS[:6]          # plan C 가 쓰는 6칸
    return c


@torch.no_grad()
def predict_studies(nets: list[M.KneeModel], studies: list[str], cfg: InferConfig,
                    ccfgs: list[cache_mod.CacheConfig], device: torch.device,
                    train: bool = False, log_every: int = 50) -> np.ndarray:
    """여러 모델의 예측을 rank 평균으로 합친다. (N, 12)

    ccfgs[i] 는 nets[i] 의 전처리 설정이다.
    같은 설정끼리 묶어 디코드를 한 번만 한다.
    """
    from scipy.stats import rankdata

    if len(nets) != len(ccfgs):
        raise ValueError(f"모델 {len(nets)}개, 설정 {len(ccfgs)}개 - 개수가 다르다")

    n_out = len(ds.resolve_plan(cfg.plan))

    # (out_px, n_slices) 가 같은 모델끼리 묶는다
    groups: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(ccfgs):
        groups.setdefault((c.out_px, c.n_slices), []).append(i)
    print(f"전처리 그룹 {len(groups)}개: "
          + ", ".join(f"{k[0]}px/{k[1]}슬 x{len(v)}모델" for k, v in groups.items()))

    preds: list[np.ndarray | None] = [None] * len(nets)
    t0 = time.time()

    for (px, nsl), idxs in groups.items():
        ccfg = ccfgs[idxs[0]]
        part = [[] for _ in idxs]

        for start in range(0, len(studies), cfg.batch_size):
            chunk = studies[start:start + cfg.batch_size]
            imgs, masks = [], []
            for u in chunk:
                try:
                    a, m = build_study_tensor(u, cfg.plan, ccfg, train=train)
                except Exception as e:                   # 손상 DICOM 등
                    print(f"  ! {u[-12:]}: {type(e).__name__}: {e}")
                    a = np.zeros((n_out, nsl, px, px), np.uint8)
                    m = np.zeros(n_out, np.float32)
                imgs.append(a); masks.append(m)

            img = torch.from_numpy(np.stack(imgs)).to(device, non_blocking=True)
            msk = torch.from_numpy(np.stack(masks)).to(device, non_blocking=True)

            for j, mi in enumerate(idxs):
                net = nets[mi]
                with torch.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                    ng = cfg.n_groups if cfg.n_groups is not None else net.n_groups
                    out = sum(net(img, msk, group=g) for g in range(ng)) / ng
                    if cfg.tta_slice_flip:
                        flip = sum(net(torch.flip(img, dims=[2]), msk, group=g)
                                   for g in range(ng)) / ng
                        out = (out + flip) / 2
                part[j].append(torch.sigmoid(out.float()).cpu().numpy())

            done = start + len(chunk)
            if log_every and (done % log_every < cfg.batch_size or done == len(studies)):
                el = time.time() - t0
                print(f"  [{px}px] {done}/{len(studies)}  {el:.0f}s", flush=True)

        for j, mi in enumerate(idxs):
            preds[mi] = np.concatenate(part[j])

    stacked = [p for p in preds if p is not None]
    if len(stacked) == 1:
        return stacked[0]
    # rank 평균 - 지표가 순서만 보므로 확률 평균보다 맞다
    ranks = [np.apply_along_axis(rankdata, 0, s) / len(s) for s in stacked]
    return np.mean(ranks, axis=0)


def write_submission(studies: list[str], probs: np.ndarray,
                     path: Path | None = None) -> pd.DataFrame:
    """제출 파일을 쓰고 형식을 검증한다."""
    path = Path(path or paths.SUBMISSION_CSV)
    sub = config.make_submission(studies, probs, path=path)

    ss = pd.read_csv(paths.SAMPLE_SUBMISSION_CSV)
    checks = {
        "컬럼 순서 일치": list(sub.columns) == list(ss.columns),
        "결측 없음": not sub[list(config.LABELS)].isna().any().any(),
        "0~1 범위": bool(((sub[list(config.LABELS)] >= 0)
                        & (sub[list(config.LABELS)] <= 1)).all().all()),
        "스터디 중복 없음": sub[config.ID_COL].is_unique,
    }
    for k, v in checks.items():
        print(f"  {'OK  ' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        raise AssertionError("제출 형식 검증 실패")
    print(f"wrote {path}  ({len(sub)} rows)")
    return sub
