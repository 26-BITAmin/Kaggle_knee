"""픽셀 캐시 빌드 - DICOM 500GB 를 모델 입력 배열로 한 번에 변환한다.

스터디 하나가 (6 슬롯, 9 슬라이스, 224, 224) uint8 배열이 된다.
4,407 스터디 전체가 약 11 GiB 다.

    from cache import build_study, CacheConfig
    arr, info = build_study(study_uid, series_rows, cfg)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import dicom_io as D
import paths


@dataclass
class CacheConfig:
    """캐시 구성.

    커널 출력 한도가 20GB 다. 4,407 스터디 기준

        224px x 8칸 x 9슬  14.8G   mm/px 0.5804   <- 1mm 구조 Nyquist 한계 초과
        280px x 6칸 x 9슬  17.4G   mm/px 0.4643
        336px x 6칸 x 6슬  16.7G   mm/px 0.3869   <- 루트와 같은 해상도

    루트 노트북은 캐시를 RAM 에 만들고 버리므로 한도가 없다. 대신 매 실행 55분을
    디코드에 쓴다. 그리고 예산이 빠듯하면 해상도가 아니라 슬라이스 수를 줄인다

      > resolution the expensive one, so when the budget binds it is the slice
      > count that gives way rather than the pixel grid

    우리는 캐시를 파일로 재사용하므로 한도 안에서 같은 선택을 한다.
    해상도가 DINOv2 패치 14 의 배수여야 한다 (336/14 = 24).
    """
    slots: tuple[str, ...] = D.SLOTS
    n_slices: int = 9
    out_px: int = 224
    extent_mm: float = 130.0
    # 중앙 20~80% 구간에서 연속 n_slices 장을 고른다.
    # 등간격 샘플링보다 연속이 낫다는 측정이 있다 (디스커션 737597)
    center_lo: float = 0.2
    center_hi: float = 0.8


@dataclass
class StudyInfo:
    study_uid: str
    slots_filled: dict = field(default_factory=dict)   # slot -> series_uid
    crop_applied: dict = field(default_factory=dict)   # slot -> bool
    mm_per_px: dict = field(default_factory=dict)      # slot -> float
    errors: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n_filled(self) -> int:
        return len(self.slots_filled)


def _read_header(path: Path):
    import pydicom

    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def classify_series(series_dir: Path) -> tuple[str | None, dict]:
    """시리즈의 첫 파일 헤더로 슬롯을 판별한다."""
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return None, {}
    ds = _read_header(files[0])
    meta = {
        "n_files": len(files),
        "plane": D.plane(ds),
        "weighting": D.weighting(ds),
        "fat_sat": D.fat_suppressed(ds),
        "spacing": D.pixel_spacing(ds),
    }
    return D.slot_of(ds), meta


def pick_series(study_dir: Path, cfg: CacheConfig) -> dict[str, Path]:
    """슬롯마다 시리즈 하나를 고른다.

    후보가 여럿이면 슬라이스가 많은 쪽, 같으면 픽셀 피치가 작은(고해상도) 쪽.
    """
    best: dict[str, tuple] = {}
    for sd in sorted(study_dir.iterdir()):
        if not sd.is_dir():
            continue
        try:
            slot, meta = classify_series(sd)
        except Exception:  # noqa: BLE001
            continue
        if slot is None or slot not in cfg.slots:
            continue
        score = (meta["n_files"], -(meta["spacing"] or 9.9))
        if slot not in best or score > best[slot][0]:
            best[slot] = (score, sd)
    return {k: v[1] for k, v in best.items()}


def read_slot(series_dir: Path, cfg: CacheConfig) -> tuple[np.ndarray, bool, float]:
    """한 시리즈에서 (n_slices, out_px, out_px) uint8 을 만든다."""
    import pydicom

    files = list(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"빈 시리즈: {series_dir}")

    # 1) 기하로 정렬한다. 파일명 정렬은 순서가 아니다.
    keyed = []
    for f in files:
        try:
            keyed.append((D.slice_key(_read_header(f)), f))
        except Exception:  # noqa: BLE001
            continue
    if not keyed:
        raise ValueError(f"헤더를 못 읽음: {series_dir}")
    keyed.sort(key=lambda t: t[0])

    # 2) 중앙 구간에서 연속 n 장
    n = len(keyed)
    lo, hi = int(n * cfg.center_lo), int(n * cfg.center_hi)
    span = max(hi - lo, 1)
    start = lo + max(0, (span - cfg.n_slices) // 2)
    chosen = [f for _, f in keyed[start:start + cfg.n_slices]]
    while len(chosen) < cfg.n_slices:           # 슬라이스가 모자라면 끝을 반복
        chosen.append(chosen[-1] if chosen else keyed[0][1])

    # 3) 픽셀을 읽고 물리 크롭
    planes, applied, mmpp = [], True, 0.0
    for f in chosen:
        ds = pydicom.dcmread(str(f), force=True)
        img = ds.pixel_array
        if img.ndim == 3:                        # multi-frame 은 가운데 프레임
            img = img[img.shape[0] // 2]
        r = D.physical_crop(img, D.pixel_spacing(ds), cfg.extent_mm, cfg.out_px)
        planes.append(r.image)
        applied &= r.applied
        mmpp = r.mm_per_px

    stack = np.stack(planes)
    return D.normalise_stack(stack), applied, mmpp


def build_study(study_uid: str, cfg: CacheConfig, train: bool = True
                ) -> tuple[np.ndarray, StudyInfo]:
    """스터디 하나를 (6, 9, 224, 224) uint8 로 만든다.

    슬롯을 못 채우면 그 자리는 0 으로 둔다. 어떤 슬롯이 비었는지는 info 에 남는다.
    """
    t0 = time.time()
    info = StudyInfo(study_uid=study_uid)
    root = paths.TRAIN_SERIES_DIR if train else paths.TEST_SERIES_DIR
    study_dir = root / study_uid

    arr = np.zeros((len(cfg.slots), cfg.n_slices, cfg.out_px, cfg.out_px), np.uint8)
    if not study_dir.exists():
        info.errors.append(f"스터디 디렉터리 없음: {study_dir}")
        info.seconds = time.time() - t0
        return arr, info

    picked = pick_series(study_dir, cfg)
    for i, slot in enumerate(cfg.slots):
        sd = picked.get(slot)
        if sd is None:
            continue
        try:
            stack, applied, mmpp = read_slot(sd, cfg)
        except Exception as e:  # noqa: BLE001  손상 DICOM 등
            info.errors.append(f"{slot}: {type(e).__name__}: {e}")
            continue
        arr[i] = stack
        info.slots_filled[slot] = sd.name
        info.crop_applied[slot] = applied
        info.mm_per_px[slot] = mmpp

    info.seconds = time.time() - t0
    return arr, info


def estimate_size(n_studies: int, cfg: CacheConfig) -> str:
    b = n_studies * len(cfg.slots) * cfg.n_slices * cfg.out_px * cfg.out_px
    return f"{b / 1024**3:.2f} GiB"
