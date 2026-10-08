"""BEATs チェックポイントの読み込みと埋め込み出力の確認。"""

import numpy as np

from egaudio.config.loader import load_yaml
from egaudio.embedding.factory import build_audio_embedder

for name in ["beats_iter3", "beats_iter3_l6", "beats_iter3_plus_as2m", "beats_iter3_plus_as2m_ft"]:
    cfg = load_yaml(f"configs/model/{name}.yaml")
    emb = build_audio_embedder(cfg)
    rng = np.random.default_rng(0)
    wavs = [rng.standard_normal(16000 * 10).astype(np.float32) * 0.1 for _ in range(3)]
    wavs.append(rng.standard_normal(16000 * 2).astype(np.float32) * 0.1)
    e = emb.encode_audio(wavs)
    print(f"{name}: finetuned={emb.finetuned} layer_idx={emb.layer} shape={e.shape} "
          f"finite={np.isfinite(e).all()} norm={np.linalg.norm(e, axis=1).round(2)}")
