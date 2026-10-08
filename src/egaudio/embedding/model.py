"""EmbeddingGemma 2 (テキスト＋音声) のラッパ。"""

from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

_ALLOWED_DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32}


def resolve_dtype(name: str) -> torch.dtype:
    """bf16 / fp32 のみ許可。fp16 は NaN・劣化出力が出るため禁止。"""
    if name not in _ALLOWED_DTYPES:
        raise ValueError(f"dtype '{name}' is not allowed (use bfloat16 or float32)")
    return _ALLOWED_DTYPES[name]


class AudioTextEmbedder:
    """テキストと音声を同一空間に埋め込む。出力は L2 正規化済み float32。"""

    def __init__(self, cfg: dict):
        from sentence_transformers import SentenceTransformer

        dtype = resolve_dtype(cfg["dtype"])
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            dtype = torch.float32
        self.sample_rate = cfg["sample_rate"]
        self.batch_size = cfg["batch_size"]
        self.model = SentenceTransformer(
            cfg["model_id"],
            device=cfg["device"],
            config_kwargs=cfg.get("config_kwargs") or {},
            model_kwargs={"torch_dtype": dtype},
        )

    def _audio_input(self, wav: np.ndarray) -> dict:
        return {"array": np.asarray(wav, dtype=np.float32), "sampling_rate": self.sample_rate}

    def _encode(self, inputs: list, batch_size: int, desc: str) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(inputs), batch_size), desc=desc, disable=len(inputs) <= batch_size):
            emb = self.model.encode(inputs[i : i + batch_size], convert_to_numpy=True, normalize_embeddings=True)
            out.append(np.asarray(emb, dtype=np.float32))
        emb = np.concatenate(out, axis=0)
        if not np.isfinite(emb).all():
            raise FloatingPointError("non-finite values in embeddings")
        return emb

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts), self.batch_size["text"], "text")

    def encode_audio(self, wavs: Sequence[np.ndarray]) -> np.ndarray:
        return self._encode([self._audio_input(w) for w in wavs], self.batch_size["audio"], "audio")
