"""EmbeddingGemma 2 の音声入力形式を調べる使い捨てプローブ。"""

import traceback

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "google/embeddinggemma-2",
    config_kwargs={"vision_config": None},
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
print(model)
print("modalities:", getattr(model, "modalities", None))
print("prompts:", model.prompts)

sr = 16000
t = np.arange(sr * 2) / sr
wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

candidates = {
    "dict_array_sr": {"array": wav, "sampling_rate": sr},
    "dict_audio_dict": {"audio": {"array": wav, "sampling_rate": sr}},
    "dict_audio_array": {"audio": wav},
    "ndarray": wav,
    "tensor": torch.from_numpy(wav),
}
for name, inp in candidates.items():
    try:
        e = model.encode([inp, inp], convert_to_numpy=True)
        print(f"OK {name}: shape={e.shape} norm={np.linalg.norm(e, axis=1)} nan={np.isnan(e).any()}")
    except Exception as ex:  # noqa: BLE001
        print(f"NG {name}: {type(ex).__name__}: {ex}")
        traceback.print_exc(limit=2)
