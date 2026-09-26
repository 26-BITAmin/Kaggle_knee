"""dicom_io 단위 테스트 - DICOM 없이 돈다.

    python tests/test_dicom_io.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import dicom_io as D

class DS:
    def __init__(self, **kw): self.__dict__.update(kw)

ok = fail = 0
def check(name, got, want):
    global ok, fail
    if got == want: ok += 1; print(f"  OK   {name}: {got}")
    else: fail += 1; print(f"  FAIL {name}: got={got} want={want}")

print("=== 평면 판별 ===")
check("sagittal", D.plane(DS(ImageOrientationPatient=[0,1,0,0,0,-1])), "Sagittal")
check("coronal",  D.plane(DS(ImageOrientationPatient=[1,0,0,0,0,-1])), "Coronal")
check("axial",    D.plane(DS(ImageOrientationPatient=[1,0,0,0,1,0])),  "Axial")

print("=== 가중 판별 ===")
check("이름에 T1",      D.weighting(DS(SeriesDescription="COR T1 TSE")), "T1")
check("이름에 PD FS",   D.weighting(DS(SeriesDescription="SAG PD FS")), "PD")
check("TR짧음->T1",     D.weighting(DS(SeriesDescription="knee", RepetitionTime=600, EchoTime=12)), "T1")
check("TR길고TE길->T2", D.weighting(DS(SeriesDescription="knee", RepetitionTime=3000, EchoTime=80)), "T2")
check("TR길고TE짧->PD", D.weighting(DS(SeriesDescription="knee", RepetitionTime=3000, EchoTime=25)), "PD")

print("=== 지방억제 ===")
check("FS",   D.fat_suppressed(DS(SeriesDescription="SAG PD FS")), True)
check("STIR", D.fat_suppressed(DS(SeriesDescription="COR STIR")), True)
check("없음", D.fat_suppressed(DS(SeriesDescription="SAG T1 TSE")), False)

print("=== 슬롯 배정 ===")
check("SAG_FLUID_FS", D.slot_of(DS(SeriesDescription="SAG PD FS", ImageOrientationPatient=[0,1,0,0,0,-1])), "SAG_FLUID_FS")
check("COR_T1",       D.slot_of(DS(SeriesDescription="COR T1", ImageOrientationPatient=[1,0,0,0,0,-1])), "COR_T1")
check("해당없음",      D.slot_of(DS(SeriesDescription="AX T1", ImageOrientationPatient=[1,0,0,0,1,0])), None)

print("=== 슬라이스 순서 (핵심) ===")
iop = [0,1,0,0,0,-1]   # sagittal, 법선 = -x
sl = [DS(ImageOrientationPatient=iop, ImagePositionPatient=[z,0,0]) for z in [5,1,3,2,4]]
keys = [D.slice_key(s) for s in sl]
order = [i for i,_ in sorted(enumerate(keys), key=lambda t: t[1])]
check("기하로 정렬", order, [0,4,2,3,1])   # z=5,4,3,2,1 (법선이 -x라 역순)
check("폴백 InstanceNumber", D.slice_key(DS(InstanceNumber=7)), 7.0)

print("=== 물리 크롭 ===")
img = np.zeros((512,512), np.uint16); img[200:312,200:312] = 1000
r = D.physical_crop(img, spacing_mm=0.3, extent_mm=130, out_px=224)
check("적용됨", r.applied, True)
check("mm/px", round(r.mm_per_px,3), round(130/224,3))
check("출력크기", r.image.shape, (224,224))
# 크롭이 원본보다 큰 경우 - 조용히 넘어가면 안 된다
r2 = D.physical_crop(np.zeros((128,128), np.uint16), spacing_mm=0.9, extent_mm=130, out_px=224)
check("크롭 불가 감지", r2.applied, False)

print("=== 정규화 ===")
st = np.concatenate([np.full((2,4,4),100), np.full((2,4,4),200)]).astype(np.uint16)
st[0,0,0] = 60000   # 이상치
n = D.normalise_stack(st)
check("uint8", n.dtype == np.uint8, True)
check("이상치에 안 뭉개짐", bool(n[2:].mean() > n[1:2].mean()), True)

print(f"\n통과 {ok} / 실패 {fail}")
sys.exit(1 if fail else 0)
