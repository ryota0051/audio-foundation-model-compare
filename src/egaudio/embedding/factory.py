"""モデル設定の `backend` から音声埋め込みモデルを生成する。"""


def build_audio_embedder(cfg: dict):
    backend = cfg.get("backend", "embeddinggemma2")
    if backend == "embeddinggemma2":
        from egaudio.embedding.model import AudioTextEmbedder

        return AudioTextEmbedder(cfg)
    if backend == "beats":
        from egaudio.embedding.beats import BeatsEmbedder

        return BeatsEmbedder(cfg)
    if backend == "clap":
        from egaudio.embedding.clap import ClapEmbedder

        return ClapEmbedder(cfg)
    raise ValueError(f"unknown backend: {backend}")
