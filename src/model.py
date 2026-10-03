"""모델 - 슬롯별 인코딩 + 레이블 쿼리 집계.

입력 한 스터디는 (n_slots, 9, 224, 224) uint8 이다.

설계 근거 (docs/baseline.md)

- 인코더는 DINOv2-S/14 를 쓰고 키우지 않는다
  Small -> Base 는 CV +0.0011, 노이즈 바닥 0.0020 으로 null 이었다 (디스커션 735154)
- 인접 3슬라이스를 3채널로 묶는다. 9장이면 그룹 3개가 된다
- 슬롯마다 임베딩을 만들고, **12개 학습 가능한 레이블 쿼리**가
  슬롯들을 가로질러 attention 한다 (디스커션 740610 의 구조)
- 빈 슬롯은 attention 에서 마스킹한다. 0 이미지를 학습시키지 않는다
- 정규화 상수는 체크포인트가 들고 다니는 계약이다 (디스커션 735154)
  timm 의 default_cfg 에서 읽어 쓴다. 직접 박아넣지 않는다
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# 어떤 소견을 어느 슬롯에서 봐야 하는가 - 해부학 지식을 공짜로 넣는다
# 루트 노트북(0.891)의 SLOT_PRIOR_TABLE 과 같은 구성이다
#
# 슬롯 인덱스 (plan A 기준, plan C 도 역할이 같아 그대로 쓴다)
#   0 SAG_FLUID_FS  1 COR_FLUID_FS  2 AX_FLUID_FS
#   3 SAG_FLUID_NOFS(=FLUID_NOFS)  4 COR_T1(=T1_1)  5 SAG_T1(=T1_2)
#
# 십자인대는 비스듬해 시상면, 측부인대와 반월상 몸통은 관상면,
# 슬개대퇴는 축상면에서 잘 보인다 (docs/competition.md)
SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5), "Effusion": (0, 2), "Synovitis": (0, 2),
    "Baker's": (0,), "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}
SLOT_PRIOR_STRENGTH = 0.55


def build_slot_prior(labels, n_slots: int, strength: float = SLOT_PRIOR_STRENGTH):
    """(n_labels, n_slots) attention 편향. 학습 가능한 초기값으로 쓴다."""
    m = torch.zeros(len(labels), n_slots)
    for i, name in enumerate(labels):
        for k in SLOT_PRIOR_TABLE.get(name, ()):
            if k < n_slots:
                m[i, k] = strength
    return m


@dataclass
class ModelConfig:
    backbone: str = "vit_small_patch14_dinov2.lvd142m"
    n_slots: int = 6
    n_slices: int = 9
    slices_per_group: int = 3        # 인접 3장을 3채널로
    n_labels: int = 12
    img_size: int = 224              # DINOv2 기본은 518 이다. 캐시가 224 라 맞춰야 한다
    dropout: float = 0.2             # 루트와 같은 값
    head_hidden: int = 256           # 384 -> 256 축소
    query_heads: int = 8             # 미사용 (SlotHead 로 교체)
    slot_prior: bool = True          # 해부학 사전 지식으로 attention 을 초기 편향
    # DINOv2-S 는 블록 12개. 마지막 몇 개만 연다
    # 전체(99)를 열고 lr 1e-4 로 돌렸더니 사전학습 특징이 파괴됐다
    unfreeze_blocks: int = 6
    pretrained: bool = True
    weights_path: str | None = None  # 오프라인 체크포인트 (커널은 인터넷이 없다)
    # 그룹(인접 3장 묶음) 처리
    #   "random" - 학습은 한 그룹만 무작위, 추론은 그룹별 예측 평균 (기존)
    #              그룹들이 서로 붙어 있을 때(center 캐시)만 맞다
    #   "tokens" - 모든 그룹을 인코딩해 (슬롯 x 그룹) 토큰으로 head 에 넣는다
    #              그룹이 떨어져 있으면(spread 캐시) 병변이 한 그룹에만 보이므로
    #              무작위 한 그룹 학습은 레이블과 어긋나고 평균은 신호를 희석한다
    group_mode: str = "random"
    # 이 모델이 학습된 캐시의 전처리 계약. 추론이 같은 입력을 다시 만들 때 쓴다
    # 기본값은 이 필드가 생기기 전 캐시(224/336 center)와 같다
    extent_mm: float = 130.0
    slice_layout: str = "center"
    triplet_centers: tuple[float, ...] = (0.25, 0.5, 0.75)


class Encoder(nn.Module):
    """timm 백본. 전처리 상수를 체크포인트에서 가져온다."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained and cfg.weights_path is None,
            num_classes=0, in_chans=cfg.slices_per_group, img_size=cfg.img_size)
        # 224/14 = 16 토큰. 해상도를 바꾸면 위치 임베딩이 보간되므로
        # 체크포인트를 바꿀 때 img_size 도 같이 확인할 것

        if cfg.weights_path:
            # 커널은 인터넷이 없다. 미리 올려둔 체크포인트를 쓴다.
            #
            # 체크포인트는 **이 img_size 로 재샘플링된 것**이어야 한다.
            # DINOv2 원본은 518 기준이라 pos_embed 가 (1, 1370, 384) 이고
            # 224 모델의 (1, 257, 384) 와 맞지 않아 RuntimeError 가 난다.
            # scripts/prepare_weights.py 가 재샘플링해 저장한다.
            #
            # strict=True 를 쓴다. strict=False 로 두면 키가 빠져도 넘어가서
            # 사전학습 없이 학습하는 상태를 눈치채지 못한다.
            sd = torch.load(cfg.weights_path, map_location="cpu")
            sd = sd.get("state_dict", sd.get("model", sd))
            try:
                self.backbone.load_state_dict(sd, strict=True)
            except RuntimeError as e:
                raise RuntimeError(
                    f"체크포인트가 img_size={cfg.img_size} 모델과 맞지 않는다.\n"
                    f"scripts/prepare_weights.py 로 다시 만들 것.\n원본 오류: {e}") from e
            print(f"[model] 오프라인 가중치 로드 {Path(cfg.weights_path).name}")

        dc = self.backbone.default_cfg
        mean = torch.tensor(dc["mean"], dtype=torch.float32)
        std = torch.tensor(dc["std"], dtype=torch.float32)
        # in_chans 를 바꿨으므로 채널 수에 맞춘다
        c = cfg.slices_per_group
        mean = mean.mean().repeat(c) if len(mean) != c else mean
        std = std.mean().repeat(c) if len(std) != c else std
        self.register_buffer("mean", mean.view(1, c, 1, 1))
        self.register_buffer("std", std.view(1, c, 1, 1))

        self.dim = self.backbone.num_features
        self._freeze(cfg.unfreeze_blocks)

    def _freeze(self, keep_last: int) -> None:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or keep_last >= len(blocks):
            return
        for p in self.backbone.parameters():
            p.requires_grad = False
        for b in blocks[-keep_last:]:
            for p in b.parameters():
                p.requires_grad = True
        for name in ("norm", "fc_norm", "head"):
            m = getattr(self.backbone, name, None)
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = True

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) uint8 -> (B, D). 정규화는 여기 GPU 에서 한다."""
        x = x_uint8.float().div_(255.0)
        x = (x - self.mean) / self.std
        return self.backbone(x)


class SlotHead(nn.Module):
    """소견별 attention 으로 슬롯 임베딩을 모은다.

    루트 노트북(0.891)의 SlotHead 와 같은 구조다. 그 주석이 왜 단순해야 하는지 설명한다

    > The aggregation is deliberately this simple. With a study-level label there is
    > no signal telling the model which part of a study matters, so extra attention
    > parameters below the slot level would have nothing to learn from and would
    > spend their capacity fitting noise.

    우리가 처음 쓴 nn.MultiheadAttention 은 Q/K/V 투영 3개로 토큰 6개를 봤다
    스터디 단위 레이블뿐이라 그걸 배울 신호가 없고, 실제로 학습이 불안정했다
    (loss 가 오르내리고 val_auc 가 0.54~0.70 을 오감)

    slot_prior 는 **고정 buffer** 다. 해부학은 데이터가 뒤집을 대상이 아니다.
    exp(0.55) ~ 1.73 이라 선호 슬롯에 1.73배 가중이 붙되 제외하지는 않는다.
    """

    def __init__(self, dim: int, n_slot: int, n_out: int, hidden: int = 256,
                 dropout: float = 0.2, prior: torch.Tensor | None = None):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        self.register_buffer("slot_prior",
                             prior if prior is not None else torch.zeros(n_out, n_slot))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (B, S, D), mask (B, S) 1=유효  ->  (B, n_out)"""
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class KneeModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        d = self.encoder.dim

        if cfg.group_mode not in ("random", "tokens"):
            raise ValueError(f"알 수 없는 group_mode: {cfg.group_mode}")
        self.n_views = cfg.n_slices // cfg.slices_per_group
        # predict 가 돌릴 그룹 수. tokens 는 한 번에 전부 보므로 1
        self.n_groups = self.n_views if cfg.group_mode == "random" else 1
        n_tok = cfg.n_slots * (self.n_views if cfg.group_mode == "tokens" else 1)
        prior = None
        if cfg.slot_prior:
            import config as _cfg
            prior = build_slot_prior(list(_cfg.LABELS), cfg.n_slots)
            if cfg.group_mode == "tokens":
                # 토큰 순서는 (슬롯, 그룹) - 같은 슬롯의 그룹들이 같은 해부학 편향을 받는다
                prior = prior.repeat_interleave(self.n_views, dim=1)
        # slot_emb 은 SlotHead 안에 있다 (루트와 같은 위치)
        # tokens 모드에서는 (슬롯, 그룹)마다 임베딩이 하나씩이라 위치 정보가 된다
        self.head = SlotHead(d, n_tok, cfg.n_labels,
                             hidden=cfg.head_hidden, dropout=cfg.dropout, prior=prior)

    def forward(self, image: torch.Tensor, slot_mask: torch.Tensor,
                group: int | None = None) -> torch.Tensor:
        """image (B, S, 9, H, W) uint8, slot_mask (B, S) -> (B, 12) 로짓

        9슬라이스를 3장씩 3그룹으로 나눈 뒤 **한 그룹만** 인코더에 넣는다.

        학습에서는 매 스텝 무작위로 고른다. 그 자체가 증강이 되고
        스터디당 인코더 입력이 18장에서 6장으로 줄어 3배 빨라진다 (루트와 같다)

        추론에서는 group 을 지정해 여러 그룹을 돌리고 평균할 수 있다.
        """
        b, s, n, h, w = image.shape
        g, c = self.n_views, self.cfg.slices_per_group

        x = image.view(b, s, g, c, h, w)
        if self.cfg.group_mode == "tokens":
            # group 인자는 무시한다. 모든 그룹이 토큰이 된다
            e = self.encoder(x.reshape(b * s * g, c, h, w)).view(b, s * g, -1)
            m = slot_mask.repeat_interleave(g, dim=1)            # (B, S*G)
            return self.head(e * m.unsqueeze(-1), m)
        if group is None:
            group = int(torch.randint(g, (1,)).item())
        if not 0 <= group < g:
            raise IndexError(
                f"group {group} 이 범위 밖이다. 이 모델은 그룹이 {g}개다 "
                f"(n_slices={n}, slices_per_group={c}). 캐시 슬라이스 수를 확인할 것")
        x = x[:, :, group]                                    # (B, S, C, H, W)

        e = self.encoder(x.reshape(b * s, c, h, w)).view(b, s, -1)   # (B, S, D)
        tok = e * slot_mask.unsqueeze(-1)
        return self.head(tok, slot_mask)


def masked_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """판독문 신뢰도를 가중으로 쓰는 soft BCE.

    **가중평균이 아니라 단순 평균이다** (루트 노트북과 같다)

        (BCE * w).mean()        <- 가중이 낮은 셀은 기여 자체가 작아진다
        (BCE * w).sum() / w.sum()  <- 정규화하면 저가중 셀의 상대적 영향이 되살아난다

    침묵 셀이 전체의 19% 인데 정규화하면 그 노이즈가 다시 커진다.
    """
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * w).mean()
