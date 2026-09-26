"""DICOM 읽기 - 조용히 틀리는 지점들을 처리한다.

세 가지가 에러 없이 망가질 수 있다 (docs/baseline.md §4, §5 참고)

1. 파일명 정렬은 물리 순서와 무관하다 (순위상관 0.01)
   ImageOrientationPatient x ImagePositionPatient 로 복원한다
2. 물리 크롭이 원본보다 크면 조용히 무시된다
   크롭이 실제로 적용됐는지 호출자가 알 수 있어야 한다
3. train_series.csv 의 Fluid_Sensitive / Fat_Suppression 이 동일하다
   TR/TE 와 시퀀스 태그에서 다시 판별한다
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 시퀀스 판별
# ---------------------------------------------------------------------------
_FS_RX = re.compile(
    r"\b(fs|fat[\s_-]*sat|fatsat|spair|spir|stir|spectral|chemsat|dixon|frfse[\s_-]*fs)\b",
    re.I,
)
_T1_RX = re.compile(r"\b(t1|t1w|t1_|tfl|mprage)\b", re.I)
_T2_RX = re.compile(r"\b(t2|t2w|t2_|stir|tirm|flair)\b", re.I)
_PD_RX = re.compile(r"\b(pd|pdw|pd_|proton|dp|dens)\b", re.I)


def _text(ds, *tags: str) -> str:
    return " ".join(str(getattr(ds, t, "") or "") for t in tags)


def weighting(ds) -> str:
    """T1 / T2 / PD 를 판별한다.

    프로토콜 이름에 적혀 있으면 그걸 쓰고, 없으면 TR/TE 로 판단한다.
    gradient echo 는 TR 이 짧게 설계돼 규칙이 깨지므로 먼저 거른다.
    """
    desc = _text(ds, "SeriesDescription", "SequenceName", "ProtocolName", "ScanOptions")

    seq = str(getattr(ds, "ScanningSequence", "") or "")
    is_gre = "GR" in seq

    if _T1_RX.search(desc):
        return "T1"
    if _PD_RX.search(desc):
        return "PD"
    if _T2_RX.search(desc):
        return "T2"

    tr = float(getattr(ds, "RepetitionTime", 0) or 0)
    te = float(getattr(ds, "EchoTime", 0) or 0)

    if is_gre:
        return "T2" if te >= 15 else "T1"
    if tr and tr <= 800:
        return "T1"
    if tr and te >= 60:
        return "T2"
    if tr:
        return "PD"

    # TR 이 없는 경우. TE 만으로도 유체 감응 여부는 가른다.
    # 슬롯 배정에는 T2 와 PD 의 구분이 필요 없다 (둘 다 fluid) 는 점을 이용한다.
    if te:
        return "T2" if te >= 60 else "PD"

    # 반전 회복(STIR/TIRM)은 유체 감응이다
    if "IR" in seq:
        return "T2"
    return "UNK"


def fat_suppressed(ds) -> bool:
    """지방 억제 여부. 가중과 독립적인 속성이다."""
    desc = _text(ds, "SeriesDescription", "SequenceName", "ProtocolName", "ScanOptions")
    if _FS_RX.search(desc):
        return True
    opts = str(getattr(ds, "ScanOptions", "") or "").upper()
    if "FS" in opts.split("\\") or "SAT" in opts:
        return True
    # STIR / TIRM 은 반전 회복으로 지방을 억제한다
    if re.search(r"\b(stir|tirm)\b", desc, re.I):
        return True
    # 짧은 TI 를 쓰는 반전 회복
    seq = str(getattr(ds, "ScanningSequence", "") or "")
    ti = float(getattr(ds, "InversionTime", 0) or 0)
    return "IR" in seq and 0 < ti <= 250


def plane(ds) -> str:
    """ImageOrientationPatient 의 법선으로 촬영 평면을 정한다."""
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        return "UNK"
    r = np.array(iop[:3], float)
    c = np.array(iop[3:], float)
    n = np.abs(np.cross(r, c))
    return ["Sagittal", "Coronal", "Axial"][int(np.argmax(n))]


def slot_of(ds) -> str | None:
    """8개 슬롯 중 어디에 속하는지. 해당 없으면 None.

    앞 6칸은 평면 고정이다 (A안). 뒤 2칸은 앞 6칸이 못 담는 것만 받는다.

    촬영 프로토콜이 기관마다 달라서 평면을 고정하면 빈칸이 많이 생긴다
    (T1 보유 스터디는 92.6% 인데 시상면 T1 로 좁히면 40.4%).
    EXTRA 칸을 두면 학습 시점에 평면 통합 설계(C안)를 코드로 만들 수 있다

        T1_1 = COR_T1 or SAG_T1 or T1_EXTRA

    앞 6칸과 픽셀이 겹치지 않으므로 중복 디코드가 없다.
    """
    p = plane(ds)
    w = weighting(ds)
    fs = fat_suppressed(ds)
    fluid = w in ("T2", "PD")

    if fluid:
        if fs:
            return {"Sagittal": "SAG_FLUID_FS", "Coronal": "COR_FLUID_FS",
                    "Axial": "AX_FLUID_FS"}.get(p)
        if p == "Sagittal":
            return "SAG_FLUID_NOFS"
        if p in ("Coronal", "Axial"):
            return "FLUID_NOFS_EXTRA"
        return None

    if w == "T1":
        if not fs:
            fixed = {"Coronal": "COR_T1", "Sagittal": "SAG_T1"}.get(p)
            if fixed:
                return fixed
        # 축상면 T1, 또는 지방억제 T1 - 앞 6칸에 자리가 없다
        return "T1_EXTRA" if p != "UNK" else None

    return None


# 앞 6칸: 평면 고정 (A안 그대로)
# 뒤 2칸: 앞 6칸이 못 담는 것만 (C안을 학습 시점에 구성하기 위함)
SLOTS = ("SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
         "SAG_FLUID_NOFS", "COR_T1", "SAG_T1",
         "T1_EXTRA", "FLUID_NOFS_EXTRA")

FIXED_SLOTS = SLOTS[:6]
EXTRA_SLOTS = SLOTS[6:]


def merged_t1_order() -> tuple[str, ...]:
    """C안의 T1_1 을 고를 우선순위. 보유율이 높은 순이다."""
    return ("COR_T1", "SAG_T1", "T1_EXTRA")


def merged_fluid_nofs_order() -> tuple[str, ...]:
    """C안의 FLUID_NOFS 를 고를 우선순위."""
    return ("SAG_FLUID_NOFS", "FLUID_NOFS_EXTRA")


# ---------------------------------------------------------------------------
# 슬라이스 순서
# ---------------------------------------------------------------------------
def slice_key(ds) -> float:
    """스택을 따라 단조 증가하는 물리 좌표.

    n = r_x  x  r_y  (ImageOrientationPatient)
    k = p . n        (ImagePositionPatient)

    파일명(SOP Instance UID)으로 정렬하면 안 된다. 순서가 아니다.
    기하 태그가 없을 때만 InstanceNumber 로 폴백한다.
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) == 3:
        r = np.array(iop[:3], float)
        c = np.array(iop[3:], float)
        return float(np.dot(np.array(ipp, float), np.cross(r, c)))
    return float(getattr(ds, "InstanceNumber", 0) or 0)


# ---------------------------------------------------------------------------
# 물리 크롭 + 정규화
# ---------------------------------------------------------------------------
@dataclass
class CropResult:
    image: np.ndarray
    applied: bool      # 크롭이 실제로 적용됐는지
    mm_per_px: float   # 리사이즈 후 유효 픽셀 피치


def physical_crop(img: np.ndarray, spacing_mm: float, extent_mm: float,
                  out_px: int) -> CropResult:
    """중심에서 extent_mm 만큼 잘라 out_px 로 맞춘다.

    extent_mm / spacing 이 원본보다 크면 크롭을 못 한다.
    그 경우 조용히 통과시키지 말고 applied=False 로 알린다.
    """
    from PIL import Image

    h, w = img.shape[:2]
    n = int(round(extent_mm / spacing_mm)) if spacing_mm > 0 else 0
    applied = 0 < n <= min(h, w)

    if applied:
        y0, x0 = (h - n) // 2, (w - n) // 2
        img = img[y0:y0 + n, x0:x0 + n]

    side = img.shape[0]
    out = np.asarray(
        Image.fromarray(img).resize((out_px, out_px), Image.BILINEAR))
    mm_per_px = (extent_mm if applied else side * spacing_mm) / out_px
    return CropResult(out, applied, mm_per_px)


def normalise_stack(stack: np.ndarray) -> np.ndarray:
    """스택 전체의 1/99 백분위수로 정규화해 uint8 로 만든다.

    슬라이스별이 아니라 스택 기준이어야 슬라이스 간 상대 대조가 유지된다.
    극값이 아니라 백분위수여야 밝은 혈관 하나가 전체를 압축하지 않는다.
    """
    x = stack.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        hi = float(x.max())
        lo = float(x.min())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def pixel_spacing(ds) -> float:
    """mm/pixel. PixelSpacing 이 없으면 0 을 돌려준다."""
    ps = getattr(ds, "PixelSpacing", None)
    if ps is not None and len(ps) >= 1:
        return float(ps[0])
    ip = getattr(ds, "ImagerPixelSpacing", None)
    if ip is not None and len(ip) >= 1:
        return float(ip[0])
    return 0.0
