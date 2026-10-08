"""CLAP (transformers ClapModel) による音声・テキスト埋め込み。

- 入力は 48kHz モノラル（モデル設定の sample_rate）。テキストにタスクプレフィックスは付けない
- プロセッサは max_seconds 超の音声をランダムに切り出す（rand_trunc）ため、ここで決定的に
  max_seconds 窓に分割（末尾は終端揃え）し、窓ごとの埋め込みを平均して 1 本にする
- 短い音声はプロセッサの repeatpad（繰り返して埋める）に任せる
"""

from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

from egaudio.audio.segment import segment
from egaudio.embedding.model import resolve_dtype


def chunk_audio(wav: np.ndarray, sr: int, max_seconds: float) -> list[np.ndarray]:
    return segment(wav, sr, window_sec=max_seconds, hop_sec=max_seconds)


def mean_of_chunks(emb: np.ndarray, owner: np.ndarray, n: int) -> np.ndarray:
    """窓ごとの埋め込みを元クリップ単位で平均し L2 正規化。"""
    out = np.zeros((n, emb.shape[1]), dtype=np.float32)
    np.add.at(out, owner, emb)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _features(out) -> torch.Tensor:
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class ClapEmbedder:
    def __init__(self, cfg: dict):
        from transformers import ClapModel, ClapProcessor

        dtype = resolve_dtype(cfg["dtype"])
        self.model = ClapModel.from_pretrained(cfg["model_id"], torch_dtype=dtype).to(cfg["device"]).eval()
        self.processor = ClapProcessor.from_pretrained(cfg["model_id"])
        self.device = cfg["device"]
        self.sample_rate = cfg["sample_rate"]
        self.max_seconds = cfg["max_seconds"]
        self.batch_size = cfg["batch_size"]

    @torch.inference_mode()
    def _encode_audio_chunks(self, chunks: list[np.ndarray]) -> np.ndarray:
        out = []
        bs = self.batch_size["audio"]
        for i in tqdm(range(0, len(chunks), bs), desc="clap-audio", disable=len(chunks) <= bs):
            inputs = self.processor(audio=chunks[i : i + bs], sampling_rate=self.sample_rate, return_tensors="pt")
            feats = _features(self.model.get_audio_features(**{k: v.to(self.device) for k, v in inputs.items()}))
            out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def encode_audio(self, wavs: Sequence[np.ndarray]) -> np.ndarray:
        chunks, owner = [], []
        for i, w in enumerate(wavs):
            cs = chunk_audio(np.asarray(w, dtype=np.float32), self.sample_rate, self.max_seconds)
            chunks += cs
            owner += [i] * len(cs)
        emb = mean_of_chunks(self._encode_audio_chunks(chunks), np.array(owner), len(wavs))
        if not np.isfinite(emb).all():
            raise FloatingPointError("non-finite values in embeddings")
        return emb

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        bs = self.batch_size["text"]
        texts = list(texts)
        for i in range(0, len(texts), bs):
            inputs = self.processor(text=texts[i : i + bs], return_tensors="pt", padding=True)
            feats = _features(self.model.get_text_features(**{k: v.to(self.device) for k, v in inputs.items()}))
            out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)
