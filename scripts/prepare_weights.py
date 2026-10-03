"""사전학습 가중치를 커널 해상도로 재샘플링해 저장한다.

    python scripts/prepare_weights.py [--img-size 224]
    python scripts/prepare_weights.py --model resnet50.a1_in1k --dataset knee-cnn-weights

커널은 인터넷이 없어 timm 이 체크포인트를 받지 못한다.
로컬에서 받아 Kaggle Dataset 으로 올려야 한다.

DINOv2 원본은 518 기준이라 pos_embed 가 (1, 1370, 384) 다.
224 모델은 (1, 257, 384) 를 기대하므로 그냥 넣으면 RuntimeError 가 난다.
timm 은 create_model(pretrained=True, img_size=...) 경로에서 보간해 주므로
그 결과를 저장한다.

CNN 은 위치 임베딩이 없어 해상도와 무관하다. img_size 를 넘기지 않는다
(ResNet 생성자는 img_size 인자를 받지 않아 TypeError 가 난다)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--in-chans", type=int, default=3)
    ap.add_argument("--dataset", default="knee-dinov2-weights")
    a = ap.parse_args()
    OUT = ROOT / "build" / a.dataset

    import timm

    is_vit = a.model.startswith("vit")
    kw = {"img_size": a.img_size} if is_vit else {}
    m = timm.create_model(a.model, pretrained=True, num_classes=0,
                          in_chans=a.in_chans, **kw)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{a.model.split('.')[0]}_{a.img_size}.pth"
    fp = OUT / name
    torch.save(m.state_dict(), fp)

    meta = {"model": a.model, "img_size": a.img_size, "in_chans": a.in_chans,
            "num_features": int(m.num_features),
            "mean": list(m.default_cfg["mean"]), "std": list(m.default_cfg["std"]),
            "pos_embed": list(m.pos_embed.shape) if is_vit else None, "file": name}
    # 해상도마다 파일이 하나씩 생긴다. 메타도 해상도별로 남긴다.
    # 공용 weights_meta.json 하나만 두면 마지막에 만든 것만 남아
    # 다른 해상도 체크포인트와 짝이 어긋난다
    # CNN 은 한 데이터셋에 여러 모델이 같은 해상도로 들어가므로 모델 이름으로 구분한다
    meta_name = (f"weights_meta_{a.img_size}.json" if is_vit
                 else f"weights_meta_{a.model.split('.')[0]}.json")
    (OUT / meta_name).write_text(json.dumps(meta, indent=2))

    # timm 가중치는 Apache-2.0 이다. 원 라이선스와 다르게 표기하지 않는다
    # 이미 있으면 (라이선스, 설명을 손으로 맞춘 경우) 덮어쓰지 않는다
    if not (OUT / "dataset-metadata.json").exists():
        (OUT / "dataset-metadata.json").write_text(json.dumps({
            "title": a.dataset,
            "id": f"hyejinkwak/{a.dataset}",
            "licenses": [{"name": "CC0-1.0" if is_vit else "other"}],
        }, indent=2))

    print(json.dumps(meta, indent=2))
    print(f"\n[+] {fp}  {fp.stat().st_size/1024**2:.1f} MB")
    print("\n업로드:")
    print(f"  kaggle datasets create -p {OUT.relative_to(ROOT)} --dir-mode zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
