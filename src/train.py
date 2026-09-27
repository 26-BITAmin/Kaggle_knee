"""1 fold 학습 루프.

평가는 macro AUC 다. AUC 는 순서만 보므로 캘리브레이션과 임계값은 의미가 없다.
그래서 체크포인트 선택도 loss 가 아니라 **검증 macro AUC** 로 한다.

검증 타깃이 판독문 유래라는 점을 기억할 것.
CV 는 "판독문을 얼마나 잘 맞추나" 를 재는 것이지 "무릎을 얼마나 잘 보나" 가 아니다.
gold 58건은 development panel 로 따로 본다 (학습에 쓰지 않는다).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config
import dataset as ds
import labels as lb
import model as M


@dataclass
class TrainConfig:
    fold: int = 0
    n_folds: int = 5
    seed: int = 42
    plan: str = "C"                  # 슬롯 구성 (A / C / ALL8)
    epochs: int = 8
    batch_size: int = 8
    accum: int = 2                   # 유효 배치 16
    # 인코더 lr. 1e-4 는 너무 높았다 - 24 epoch 실행에서 epoch 0 이 최고였고
    # 12개 레이블 중 10개가 학습할수록 나빠졌다 (DINOv2 특징 파괴)
    lr: float = 8e-6
    head_lr_mult: float = 125.0      # head 는 1e-3 (루트 노트북 LR_HEAD)
    weight_decay: float = 0.02
    warmup_frac: float = 0.2         # 초반 발산을 막는다
    num_workers: int = 4
    amp: bool = True
    grad_clip: float = 1.0
    debug: bool = False
    debug_studies: int = 64
    w_unaddressed: float = 0.3
    out_dir: str = "/kaggle/working"

    def __post_init__(self):
        if self.debug:
            self.epochs = 1


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, dict]:
    """레이블별 AUC 와 그 평균. 한 클래스뿐인 컬럼은 건너뛴다."""
    per = {}
    for i, c in enumerate(config.LABELS):
        yi = y[:, i]
        if len(np.unique(yi)) < 2:
            per[c] = float("nan")
            continue
        per[c] = float(roc_auc_score(yi, p[:, i]))
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def build_loaders(cache: ds.CacheHandle, cfg: TrainConfig):
    y, w = lb.load_targets(lb.LabelConfig(w_unaddressed=cfg.w_unaddressed))
    gold = lb.load_gold()

    studies = [s for s in cache.index.study if s in y.index]
    folds = lb.make_folds(pd.Index(studies), y, cfg.n_folds, cfg.seed, gold=gold.index)

    tr = [s for s in studies if folds[s] != cfg.fold]
    va = [s for s in studies if folds[s] == cfg.fold]
    if cfg.debug:
        tr, va = tr[: cfg.debug_studies], va[: cfg.debug_studies // 2]

    print(f"train {len(tr):,}  valid {len(va):,}  (fold {cfg.fold}/{cfg.n_folds})")
    gold_in_valid = len(set(va) & set(gold.index))
    print(f"검증셋 안의 gold 스터디: {gold_in_valid} / {len(gold)}")

    train_ds = ds.KneeCacheDataset(cache, tr, y, w, plan=cfg.plan,
                                   augment=True, seed=cfg.seed)
    valid_ds = ds.KneeCacheDataset(cache, va, y, w, plan=cfg.plan, augment=False)

    sampler = ds.ShardSequentialSampler(train_ds, shuffle=True, seed=cfg.seed)
    common = dict(num_workers=cfg.num_workers, pin_memory=True,
                  persistent_workers=cfg.num_workers > 0)
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                   drop_last=True, **common),
        DataLoader(valid_ds, batch_size=cfg.batch_size * 2, shuffle=False, **common),
        sampler, gold, va,
    )


def param_groups(net: M.KneeModel, cfg: TrainConfig):
    enc, head = [], []
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (enc if n.startswith("encoder.") else head).append(p)
    return [
        {"params": enc, "lr": cfg.lr},
        {"params": head, "lr": cfg.lr * cfg.head_lr_mult},
    ]


@torch.no_grad()
def predict(net, loader, device, amp: bool):
    net.eval()
    P, Y, S = [], [], []
    for b in loader:
        img = b["image"].to(device, non_blocking=True)
        mask = b["slot_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = net(img, mask)
        P.append(torch.sigmoid(out.float()).cpu().numpy())
        if "y" in b:
            Y.append(b["y"].numpy())
        S.extend(b["study"])
    return (np.concatenate(P),
            np.concatenate(Y) if Y else None,
            S)


def run(cache_root: str, cfg: TrainConfig, mcfg: M.ModelConfig | None = None) -> dict:
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}  {torch.cuda.get_device_name(0) if device.type=='cuda' else ''}")

    cache = ds.open_cache(cache_root)
    train_loader, valid_loader, sampler, gold, va_studies = build_loaders(cache, cfg)

    n_slots = len(ds.resolve_plan(cfg.plan))
    mcfg = mcfg or M.ModelConfig()
    mcfg.n_slots = n_slots
    net = M.KneeModel(mcfg).to(device)
    print(f"plan {cfg.plan}  슬롯 {n_slots}칸  "
          f"학습 파라미터 {sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.1f}M")

    opt = torch.optim.AdamW(param_groups(net, cfg), weight_decay=cfg.weight_decay)
    steps = max(1, len(train_loader) // cfg.accum) * cfg.epochs
    warm = max(1, int(steps * cfg.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    best = {"auc": -1.0, "epoch": -1}
    hist = []

    for ep in range(cfg.epochs):
        sampler.set_epoch(ep)
        net.train()
        te, run_loss, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)

        for i, b in enumerate(train_loader):
            img = b["image"].to(device, non_blocking=True)
            mask = b["slot_mask"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            w = b["w"].to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                loss = M.masked_bce(net(img, mask), y, w) / cfg.accum
            scaler.scale(loss).backward()

            if (i + 1) % cfg.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            run_loss += float(loss.detach()) * cfg.accum
            nb += 1

        p, yv, studies = predict(net, valid_loader, device, cfg.amp)
        auc, per = macro_auc((yv >= 0.5).astype(int), p)

        # gold 패널 - 검증셋에 들어온 gold 스터디만 본다
        # gold 는 58건뿐이라 fold 하나에 9~13건만 들어간다
        # 표본이 작아 신뢰구간이 매우 넓지만, 방향성은 볼 수 있으므로 기록은 한다
        gmask = [i for i, s in enumerate(studies) if s in gold.index]
        gauc = float("nan")
        if len(gmask) >= 6:
            gy = gold.loc[[studies[i] for i in gmask]].values
            gauc, _ = macro_auc(gy, p[gmask])

        row = {"epoch": ep, "loss": run_loss / max(nb, 1), "val_auc": auc,
               "gold_auc": gauc, "sec": time.time() - te}
        hist.append(row | {"per_label": per})
        print(f"ep {ep}  loss {row['loss']:.4f}  val_auc {auc:.4f}  "
              f"gold {gauc:.4f}  {row['sec']:.0f}s", flush=True)

        if auc > best["auc"]:
            best = {"auc": auc, "epoch": ep}
            torch.save({"model": net.state_dict(),
                        "train_cfg": asdict(cfg), "model_cfg": asdict(mcfg),
                        "val_auc": auc, "epoch": ep},
                       out / f"fold{cfg.fold}_best.pt")
            np.save(out / f"fold{cfg.fold}_oof.npy", p)
            pd.Series(studies, name="study").to_csv(out / f"fold{cfg.fold}_oof_studies.csv",
                                                    index=False)

    report = {"best": best, "history": hist, "minutes": (time.time() - t0) / 60,
              "train_cfg": asdict(cfg), "model_cfg": asdict(mcfg)}
    (out / f"fold{cfg.fold}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float))
    print(f"\nbest val_auc {best['auc']:.4f} (epoch {best['epoch']})  "
          f"{report['minutes']:.1f}분")
    return report
