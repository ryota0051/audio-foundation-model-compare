"""sampling_rate の扱いと音声⇔テキスト整合性の簡易確認。"""

import numpy as np
import soundfile as sf
import torch
from sentence_transformers import SentenceTransformer

from egaudio.audio.preprocess import resample

model = SentenceTransformer(
    "google/embeddinggemma-2",
    config_kwargs={"vision_config": None},
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
raw, sr = sf.read("data/ESC-50/audio/1-100032-A-0.wav", dtype="float32")  # dog
wav16 = resample(raw, sr, 16000)

inputs = [wav16, {"array": wav16, "sampling_rate": 16000}, {"array": raw, "sampling_rate": sr}, raw]
e = np.concatenate([model.encode([x], convert_to_numpy=True, normalize_embeddings=True) for x in inputs])
print("16k ndarray vs dict16k:", float(e[0] @ e[1]))
print("16k ndarray vs dict44.1k(raw):", float(e[0] @ e[2]))
print("16k ndarray vs raw44.1k ndarray:", float(e[0] @ e[3]))

texts = ["task: classification | query: a dog barking", "task: classification | query: rain falling",
         "task: classification | query: a car horn honking"]
t = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
print("audio(dog) vs texts:", (e[0] @ t.T).round(3))
