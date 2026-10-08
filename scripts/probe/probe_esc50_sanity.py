"""ESC-50 fold1 の精度が低い原因切り分け（dtype / バッチ / プロンプト）。"""

import torch
from sentence_transformers import SentenceTransformer

from egaudio.audio.preprocess import load_audio
from egaudio.config.loader import load_yaml
from egaudio.data.esc50 import load_meta
from egaudio.zeroshot.classify import cosine_similarity, topk_accuracy
from egaudio.zeroshot.prompts import build_texts

exp = load_yaml("configs/experiment/esc50_zeroshot.yaml")
meta = load_meta("data/ESC-50/meta/esc50.csv", "data/ESC-50/audio", [1])
classes = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
y = meta["target"].to_numpy()
wavs = [{"array": load_audio(p, 16000, None), "sampling_rate": 16000} for p in meta["path"]]

prefixes = {
    "none": "{text}",
    "search": "task: search result | query: {text}",
    "document": "title: none | text: {text}",
}

for dtype in [torch.bfloat16, torch.float32]:
    model = SentenceTransformer(
        "google/embeddinggemma-2", config_kwargs={"vision_config": None},
        model_kwargs={"torch_dtype": dtype}, device="cuda",
    )
    print("dtype", dtype, "default_prompt_name:", model.default_prompt_name)
    for bs in [1, 8]:
        if dtype is torch.float32 and bs == 1:
            continue
        a = model.encode(wavs, batch_size=bs, convert_to_numpy=True, normalize_embeddings=True)
        for pname, p in prefixes.items():
            t = model.encode(build_texts(classes, exp["templates"]["natural"], p, exp["natural_labels"]),
                             convert_to_numpy=True, normalize_embeddings=True)
            sim = cosine_similarity(a, t)
            print(f"  bs={bs} prefix={pname}: top1={topk_accuracy(sim, y, 1):.4f} top5={topk_accuracy(sim, y, 5):.4f}")
    del model
    torch.cuda.empty_cache()
