"""사전학습 가중치를 커널 해상도로 재샘플링해 저장한다.

    python scripts/prepare_weights.py [--img-size 224]

커널은 인터넷이 없어 timm 이 체크포인트를 받지 못한다.
로컬에서 받아 Kaggle Dataset 으로 올려야 한다.

DINOv2 원본은 518 기준이라 pos_embed 가 (1, 1370, 384) 다.
224 모델은 (1, 257, 384) 를 기대하므로 그냥 넣으면 RuntimeError 가 난다.
timm 은 create_model(pretrained=True, img_size=...) 경로에서 보간해 주므로
그 결과를 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "knee-dinov2-weights"
DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--in-chans", type=int, default=3)
    a = ap.parse_args()

    import timm

    m = timm.create_model(a.model, pretrained=True, num_classes=0,
                          in_chans=a.in_chans, img_size=a.img_size)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{a.model.split('.')[0]}_{a.img_size}.pth"
    fp = OUT / name
    torch.save(m.state_dict(), fp)

    meta = {"model": a.model, "img_size": a.img_size, "in_chans": a.in_chans,
            "num_features": int(m.num_features),
            "mean": list(m.default_cfg["mean"]), "std": list(m.default_cfg["std"]),
            "pos_embed": list(m.pos_embed.shape), "file": name}
    # 해상도마다 파일이 하나씩 생긴다. 메타도 해상도별로 남긴다.
    # 공용 weights_meta.json 하나만 두면 마지막에 만든 것만 남아
    # 다른 해상도 체크포인트와 짝이 어긋난다
    (OUT / f"weights_meta_{a.img_size}.json").write_text(json.dumps(meta, indent=2))

    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "knee-dinov2-weights",
        "id": "hyejinkwak/knee-dinov2-weights",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    print(json.dumps(meta, indent=2))
    print(f"\n[+] {fp}  {fp.stat().st_size/1024**2:.1f} MB")
    print("\n업로드:")
    print(f"  kaggle datasets create -p {OUT.relative_to(ROOT)} --dir-mode zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
