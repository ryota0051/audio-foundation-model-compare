"""BEATs (microsoft/unilm) による音声埋め込み。

指定層のエンコーダ出力を時間方向に平均プーリングしてクリップ埋め込みにする。
AudioSet ファインチューニング済みチェックポイントでも分類ヘッドは使わず、エンコーダ出力を取り出す。
"""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

from egaudio.embedding.model import resolve_dtype


def group_by_length(wavs: Sequence[np.ndarray], batch_size: int) -> list[list[int]]:
    """同じ長さの波形だけでバッチを組む（パディング不要にするため）。"""
    groups: dict[int, list[int]] = defaultdict(list)
    for i, w in enumerate(wavs):
        groups[len(w)].append(i)
    return [idx[j : j + batch_size] for idx in groups.values() for j in range(0, len(idx), batch_size)]


def resolve_layer(layer: int | None, num_layers: int) -> int:
    """1 始まりの層番号（None = 最終層）を 0 始まりに変換。"""
    if layer is None:
        return num_layers - 1
    if not 1 <= layer <= num_layers:
        raise ValueError(f"layer {layer} out of range 1..{num_layers}")
    return layer - 1


class BeatsEmbedder:
    def __init__(self, cfg: dict):
        from egaudio.embedding.beats_vendor.BEATs import BEATs, BEATsConfig

        ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
        model_cfg = BEATsConfig(ckpt["cfg"])
        self.model = BEATs(model_cfg)
        self.model.load_state_dict(ckpt["model"])
        self.device = cfg["device"]
        self.dtype = resolve_dtype(cfg["dtype"])
        self.model.to(self.device).eval()
        self.layer = resolve_layer(cfg.get("layer"), model_cfg.encoder_layers)
        self.batch_size = cfg["batch_size"]["audio"]
        self.finetuned = bool(model_cfg.finetuned_model)

    @torch.inference_mode()
    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        m = self.model
        fbank = m.preprocess(batch).unsqueeze(1)
        x = m.patch_embedding(fbank)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = m.layer_norm(x)
        if m.post_extract_proj is not None:
            x = m.post_extract_proj(x)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype is not torch.float32):
            x, _ = m.encoder.extract_features(x, padding_mask=None, tgt_layer=self.layer)
        return x.float().mean(dim=1)

    def encode_audio(self, wavs: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(wavs), self.model.cfg.encoder_embed_dim), dtype=np.float32)
        for idx in tqdm(group_by_length(wavs, self.batch_size), desc="beats", disable=len(wavs) <= self.batch_size):
            batch = torch.from_numpy(np.stack([wavs[i] for i in idx]).astype(np.float32)).to(self.device)
            out[idx] = self._forward(batch).cpu().numpy()
        if not np.isfinite(out).all():
            raise FloatingPointError("non-finite values in embeddings")
        return out
