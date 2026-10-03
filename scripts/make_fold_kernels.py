"""fold 0 학습 커널을 복제해 fold 1~4 커널을 만든다

    python scripts/make_fold_kernels.py train_coatnet
    python scripts/make_fold_kernels.py train_336 --folds 1 2

kernels/<base>_f<k>/ 가 생긴다. 푸시는 따로 한다
    bash scripts/push_kernel.sh train_coatnet_f1

fold 마다 커널을 따로 두는 이유
- CoAtNet 은 fold 하나에 3.6시간이라 4 fold 를 한 커널에 넣으면 세션 한도 12시간을 넘는다
- 커널 출력이 fold 별로 남아 제출 커널이 kernel_sources 로 골라 붙일 수 있다
- 하나가 죽어도 그 fold 만 다시 돌리면 된다

바꾸는 것은 fold 번호와 커널 id 뿐이다. 학습 설정은 fold 0 과 같아야 OOF 를 이어 붙일 수 있다
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 예전에 검증된 GPU 이미지. 기본 이미지는 Python 3.13 이라 휠 설치가 깨진다
GPU_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
             "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="kernels/ 아래 fold 0 커널 디렉터리 이름 (예: train_coatnet)")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    a = ap.parse_args()

    src_dir = ROOT / "kernels" / a.base
    meta = json.loads((src_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    nb_text = (src_dir / meta["code_file"]).read_text(encoding="utf-8")

    # 학습 설정 셀에 fold=0 이 정확히 한 번 있어야 한다
    needle = "fold=0, plan="
    if nb_text.count(needle) != 1:
        raise SystemExit(f"{a.base}: '{needle}' 가 {nb_text.count(needle)}번 나온다. 한 번이어야 한다")

    for k in a.folds:
        if not 1 <= k <= 4:
            raise SystemExit(f"fold {k} 는 1~4 여야 한다 (fold 0 은 원본 커널)")
        name = f"{a.base}_f{k}"
        out = ROOT / "kernels" / name
        out.mkdir(exist_ok=True)

        nb = json.loads(nb_text.replace(needle, f"fold={k}, plan="))
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                c["outputs"], c["execution_count"] = [], None
        code_file = f"{name}.ipynb"
        (out / code_file).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

        slug = meta["id"].split("/", 1)[1]
        m = dict(meta)
        m.update(id=f"{meta['id'].split('/')[0]}/{slug}-f{k}", title=f"{slug}-f{k}",
                 code_file=code_file, docker_image_pinning_type="original",
                 docker_image=meta.get("docker_image") or GPU_IMAGE)
        (out / "kernel-metadata.json").write_text(
            json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[+] kernels/{name}  ({m['id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
