"""transformers の CLAP API と ESC-50 1 クリップでの音声⇔テキスト整合性を確認する。"""

import inspect

import torch
from transformers import ClapModel, ClapProcessor

from egaudio.audio.preprocess import load_audio

mid = "laion/larger_clap_general"
model = ClapModel.from_pretrained(mid).to("cuda").eval()
proc = ClapProcessor.from_pretrained(mid)
print("processor __call__ params:", list(inspect.signature(proc.__call__).parameters))
print("get_audio_features params:", list(inspect.signature(model.get_audio_features).parameters))

wav = load_audio("data/ESC-50/audio/1-100032-A-0.wav", 48000, None)  # dog
texts = ["This is a sound of a dog barking.", "This is a sound of rain falling.", "This is a sound of a car horn."]
with torch.inference_mode():
    for kw in ["audio", "audios"]:
        try:
            a_in = proc(**{kw: [wav, wav[: 48000 * 2]]}, sampling_rate=48000, return_tensors="pt")
            print(f"OK kw={kw}: keys={list(a_in.keys())} shapes={[tuple(v.shape) for v in a_in.values()]}")
            break
        except Exception as ex:  # noqa: BLE001
            print(f"NG kw={kw}: {type(ex).__name__}: {ex}")
    a = model.get_audio_features(**{k: v.to("cuda") for k, v in a_in.items()})
    t_in = proc(text=texts, return_tensors="pt", padding=True)
    t = model.get_text_features(**{k: v.to("cuda") for k, v in t_in.items()})
    print("audio feat type:", type(a), "text feat type:", type(t))
    a = a if isinstance(a, torch.Tensor) else a.pooler_output
    t = t if isinstance(t, torch.Tensor) else t.pooler_output
    a = torch.nn.functional.normalize(a, dim=-1).cpu().numpy()
    t = torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
print("audio shape", a.shape, "text shape", t.shape)
print("dog clip vs texts:", (a[0] @ t.T).round(3))
