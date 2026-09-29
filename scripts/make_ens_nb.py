"""kernels/submit_ens/submit_ens.ipynb 를 생성한다

    python scripts/make_ens_nb.py

공개 가중치 앙상블 (journey.md 8장 1번)
pjmathematician/rsna-knee-d4-lite (공개 LB 0.943) 의 공개 파트(셀 0~29)를 그대로 쓰고
작성자 전용 레그(셀 30~32, 비공개 데이터셋 eff6/ens14)를 우리 모델로 바꾼다

- d4-lite 셀은 한 글자도 고치지 않는다
  체크포인트 해시, 의존성 버전, GPU 2장을 스스로 검사하는 코드라 손대면 깨진다
- 우리 레그는 서브프로세스로 돌린다
  d4-lite 가 전역에 수백 개 이름을 깔아 두고, 우리 모듈 이름(paths, config, model...)이 흔하다
- 우리 레그가 어떤 이유로든 실패하면 공개 결과 그대로 제출한다
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "refs" / "kernels" / "pjmathematician_rsna-knee-d4-lite" / "rsna-knee-d4-lite.ipynb"
OUT = ROOT / "kernels" / "submit_ens" / "submit_ens.ipynb"

# d4-lite 셀 29 가 submission.csv 를 확정한다. 그 뒤(30~)는 작성자 전용 레그
PUBLIC_LAST = 29

# 블렌드에서 우리 몫. 공개 0.943 vs 우리 0.877 이라 작게 시작한다
OURS_W = 0.15


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines]}


OUR_LEG_SCRIPT = r'''
import os, sys, time
from pathlib import Path

for _c in ['/kaggle/input/datasets/hyejinkwak/knee-src', '/kaggle/input/knee-src']:
    if os.path.isdir(_c):
        sys.path.insert(0, _c); break
else:
    raise FileNotFoundError('knee-src 없음')

import subprocess
import paths, config
config.verify_labels()

wheels = paths.find_input('knee-dicom-wheels')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index',
                '--find-links', str(wheels),
                'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'python-gdcm'],
               check=True)

import pandas as pd, torch
import infer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
studies = pd.read_csv(paths.TEST_CSV)[config.ID_COL].astype(str).tolist()

ckpts = sorted(Path(paths.find_input('knee-weights')).glob('**/*.pt'))
assert ckpts, 'knee-weights 에 .pt 가 없다'
nets, ccfgs, plans = [], [], set()
for c in ckpts:
    net, ck = infer.load_model(c, device)
    nets.append(net); ccfgs.append(infer.config_for(ck)); plans.add(ck['train_cfg']['plan'])
    print(f'  {c.name}  val_auc {ck.get("val_auc", float("nan")):.4f}', flush=True)
assert len(plans) == 1, f'plan 이 서로 다르다: {plans}'

t = time.time()
icfg = infer.InferConfig(plan=plans.pop(), batch_size=8, amp=True)
probs = infer.predict_studies(nets, studies, icfg, ccfgs, device, train=False, log_every=100)
print(f'추론 {(time.time()-t)/60:.1f}분', flush=True)

out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
infer.write_submission(studies, probs, out)
'''


OUR_CELLS = [
    md("## 우리 레그 - 224 + 336 DINOv2",
       "",
       "여기까지가 d4-lite 공개 파이프라인이다 (공개 LB 0.943)",
       "",
       "공개 결과를 먼저 보관하고 우리 모델을 서브프로세스로 돌린 뒤 rank 공간에서 섞는다",
       "",
       "우리 레그가 실패하면 공개 결과를 그대로 제출한다"),

    code("# 1/3 공개 결과 보관",
         "import os as _k_os, shutil as _k_shutil, time as _k_time",
         "from pathlib import Path as _KPath",
         "import numpy as _k_np, pandas as _k_pd",
         "",
         "_K_WORK = _KPath('/kaggle/working')",
         "_K_SUB = _K_WORK / 'submission.csv'",
         "_K_PUBLIC = _K_WORK / '_public_d4lite.csv'",
         "_K_OURS = _KPath('/kaggle/tmp/ours/submission.csv')",
         "_K_LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',",
         "             'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', \"Baker's\",",
         "             'Contusion', 'Fracture']",
         "",
         "_k_pub = _k_pd.read_csv(_K_SUB, dtype={'StudyInstanceUID': str})",
         "assert _k_pub.columns.tolist() == ['StudyInstanceUID', *_K_LABELS], _k_pub.columns.tolist()",
         "assert _k_pub[_K_LABELS].notna().all().all(), '공개 결과에 결측'",
         "_k_shutil.copy2(_K_SUB, _K_PUBLIC)",
         "print(f'[ours 1/3] 공개 결과 보관 {_k_pub.shape}  경과 {(_k_time.time()-T0)/60:.1f}분', flush=True)"),

    code("# 2/3 우리 모델 추론 (서브프로세스, GPU 0 한 장)",
         "import subprocess as _k_sp, sys as _k_sys",
         "",
         f"_K_SCRIPT = {OUR_LEG_SCRIPT!r}",
         "_k_script = _KPath('/kaggle/tmp/ours_leg.py')",
         "_k_script.parent.mkdir(parents=True, exist_ok=True)",
         "_k_script.write_text(_K_SCRIPT)",
         "",
         "# 9시간 한도에서 20분 여유를 남긴다",
         "_k_left = 9 * 3600 - (_k_time.time() - T0) - 20 * 60",
         "print(f'[ours 2/3] 남은 시간 {_k_left/60:.0f}분', flush=True)",
         "_k_rc = None",
         "if _k_left > 10 * 60:",
         "    try:",
         "        _k_rc = _k_sp.run([_k_sys.executable, str(_k_script), str(_K_OURS)],",
         "                          env={**_k_os.environ, 'CUDA_VISIBLE_DEVICES': '0'},",
         "                          timeout=_k_left).returncode",
         "    except _k_sp.TimeoutExpired:",
         "        print('[ours 2/3] 시간 초과 - 공개 결과만 쓴다', flush=True)",
         "print(f'[ours 2/3] rc={_k_rc}  경과 {(_k_time.time()-T0)/60:.1f}분', flush=True)"),

    code("# 3/3 rank 블렌드",
         "# 지표가 레이블별 순서만 보므로 rank 공간에서 섞는다",
         "# 가중치는 하나만 쓴다. 레이블별 12개는 검증할 데이터가 없다",
         f"_K_W = {OURS_W}",
         "",
         "def _k_rank(df):",
         "    return df[_K_LABELS].rank(method='average', pct=True).to_numpy()",
         "",
         "_k_blended = False",
         "try:",
         "    if not _K_OURS.is_file():",
         "        raise RuntimeError('우리 레그가 csv 를 안 썼다')",
         "    _k_ours = _k_pd.read_csv(_K_OURS, dtype={'StudyInstanceUID': str})",
         "    if _k_ours.columns.tolist() != ['StudyInstanceUID', *_K_LABELS]:",
         "        raise RuntimeError(f'컬럼 불일치 {_k_ours.columns.tolist()}')",
         "    _k_ours = _k_ours.set_index('StudyInstanceUID').reindex(_k_pub['StudyInstanceUID']).reset_index()",
         "    if _k_ours[_K_LABELS].isna().any().any():",
         "        raise RuntimeError('우리 레그에 빠진 스터디가 있다')",
         "    # 상수 컬럼은 폴백이 돌았다는 뜻이다. 공개 3건 실행에서는 정상일 수 있다",
         "    _k_sd = float(_k_ours[_K_LABELS].to_numpy(float).std(axis=0).min())",
         "    if _k_sd < 1e-9 and len(_k_pub) > 10:",
         "        raise RuntimeError(f'상수 컬럼 (min sd {_k_sd:.1e})')",
         "",
         "    _k_out = _k_pub.copy()",
         "    _k_out[_K_LABELS] = (1 - _K_W) * _k_rank(_k_pub) + _K_W * _k_rank(_k_ours)",
         "    _k_out[_K_LABELS] = _k_out[_K_LABELS].rank(method='average', pct=True)",
         "    assert _k_np.isfinite(_k_out[_K_LABELS].to_numpy(float)).all()",
         "    _k_out.to_csv(_K_SUB, index=False)",
         "    _k_blended = True",
         "",
         "    _k_corr = [_k_pd.Series(_k_rank(_k_pub)[:, j]).corr(_k_pd.Series(_k_rank(_k_ours)[:, j]), method='spearman')",
         "               for j in range(len(_K_LABELS))]",
         "    print('[ours 3/3] 레이블별 공개-우리 순위상관', flush=True)",
         "    for _n, _c in zip(_K_LABELS, _k_corr):",
         "        print(f'    {_n:18s} {_c:.3f}')",
         "    print(f'    평균 {_k_np.nanmean(_k_corr):.3f}')",
         "except Exception as _k_exc:",
         "    print(f'[ours 3/3] 블렌드 실패 ({type(_k_exc).__name__}: {_k_exc}) - 공개 결과만 쓴다', flush=True)",
         "    _k_shutil.copy2(_K_PUBLIC, _K_SUB)",
         "",
         "_k_final = _k_pd.read_csv(_K_SUB, dtype={'StudyInstanceUID': str})",
         "assert _k_final.columns.tolist() == ['StudyInstanceUID', *_K_LABELS]",
         "assert _k_final[_K_LABELS].notna().all().all()",
         "assert len(_k_final) == len(_k_pub)",
         "print(f'[ours] 최종: {\"우리 w=%.2f 블렌드\" % _K_W if _k_blended else \"공개 결과만\"}  '",
         "      f'{_k_final.shape}  총 {(_k_time.time()-T0)/60:.1f}분', flush=True)"),
]


def main():
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    public = nb["cells"][:PUBLIC_LAST + 1]
    # 마지막 공개 셀이 submission.csv 를 확정하는 셀인지 확인한다
    assert "COMPLETE: {_final}" in "".join(public[-1]["source"]), "d4-lite 셀 구성이 바뀌었다"
    for c in public:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None

    nb["cells"] = public + OUR_CELLS
    # d4-lite 셀 15 의 augment 는 jitter=False 라 도달하지 않는 분기에서만 쓰인다
    nb.setdefault("metadata", {})["check_notebook"] = {"allow_undefined": ["augment"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}  (public {len(public)} + ours {len(OUR_CELLS)} cells)")


if __name__ == "__main__":
    main()
