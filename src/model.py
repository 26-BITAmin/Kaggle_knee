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


@dataclass
class ModelConfig:
    backbone: str = "vit_small_patch14_dinov2.lvd142m"
    n_slots: int = 6
    n_slices: int = 9
    slices_per_group: int = 3        # 인접 3장을 3채널로
    n_labels: int = 12
    img_size: int = 224              # DINOv2 기본은 518 이다. 캐시가 224 라 맞춰야 한다
    dropout: float = 0.1
    query_heads: int = 8
    unfreeze_blocks: int = 99        # 기본은 전체 fine-tune (freeze 보다 낫다)
    pretrained: bool = True
    weights_path: str | None = None  # 오프라인 체크포인트 (커널은 인터넷이 없다)


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


class LabelQueryHead(nn.Module):
    """12개 레이블 쿼리가 슬롯 임베딩을 가로질러 attention 한다.

    빈 슬롯은 key_padding_mask 로 제외한다.
    슬롯이 비는 비율이 기관마다 다르므로 (docs/eda.md) 0 을 학습시키면
    빈칸 패턴 자체가 기관 힌트가 된다.
    """

    def __init__(self, dim: int, n_labels: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_labels, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Parameter(torch.zeros(n_labels, dim))
        self.bias = nn.Parameter(torch.zeros(n_labels))
        nn.init.trunc_normal_(self.out, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens (B, S, D), mask (B, S) 1=유효  ->  (B, n_labels)"""
        b = tokens.size(0)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        pad = mask < 0.5
        # 전부 빈 스터디는 없지만(검증 완료) 안전장치로 첫 칸을 열어둔다
        allempty = pad.all(dim=1)
        if allempty.any():
            pad = pad.clone()
            pad[allempty, 0] = False
        z, _ = self.attn(q, tokens, tokens, key_padding_mask=pad, need_weights=False)
        z = self.drop(self.norm(z))
        return (z * self.out.unsqueeze(0)).sum(-1) + self.bias


class KneeModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        d = self.encoder.dim

        self.n_groups = cfg.n_slices // cfg.slices_per_group
        # 슬롯 정체성을 알려준다. 같은 칸이라도 시상/관상이 섞일 수 있다 (C안)
        self.slot_emb = nn.Parameter(torch.zeros(cfg.n_slots, d))
        nn.init.trunc_normal_(self.slot_emb, std=0.02)
        self.group_pool = nn.Linear(d, 1)
        self.head = LabelQueryHead(d, cfg.n_labels, cfg.query_heads, cfg.dropout)

    def forward(self, image: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        """image (B, S, 9, H, W) uint8, slot_mask (B, S) -> (B, 12) 로짓"""
        b, s, n, h, w = image.shape
        g, c = self.n_groups, self.cfg.slices_per_group

        x = image.view(b * s * g, c, h, w)
        e = self.encoder(x).view(b, s, g, -1)                 # (B, S, G, D)

        # 그룹 3개를 attention pooling 으로 합친다
        a = torch.softmax(self.group_pool(e).squeeze(-1), dim=-1)   # (B, S, G)
        tok = (e * a.unsqueeze(-1)).sum(2)                    # (B, S, D)
        tok = tok + self.slot_emb.unsqueeze(0)
        tok = tok * slot_mask.unsqueeze(-1)

        return self.head(tok, slot_mask)


def masked_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """판독문 신뢰도를 가중으로 쓰는 soft BCE.

    미언급 셀은 w 가 낮다 (labels.LabelConfig.w_unaddressed).
    마스킹(0)이 아니라 저가중인 이유는 docs/labels.md 참고.
    """
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * w).sum() / w.sum().clamp_min(1.0)
