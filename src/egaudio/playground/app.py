"""テキストとマイク音声を埋め込み、コサイン類似度を表示する Gradio アプリ。"""

import numpy as np

from egaudio.playground.similarity import apply_prefix, parse_texts, prepare_mic_audio, rank_similarities

COLUMNS = ["rank", "text", "cosine", "diff_from_mean"]


def compute(embedder, cfg: dict, raw_texts: str, prefix_name: str, audio) -> tuple[list[list], str]:
    """UI から呼ばれる処理本体。(表の行, 補足メッセージ) を返す。"""
    texts = parse_texts(raw_texts)
    if not texts:
        raise ValueError("テキストを 1 行以上入力してください")
    wav = prepare_mic_audio(audio, target_sr=embedder.sample_rate, max_sec=cfg["audio"]["max_sec"])

    template = cfg["text_prefixes"][prefix_name]
    text_emb = embedder.encode_text([apply_prefix(t, template) for t in texts])
    audio_emb = embedder.encode_audio([wav])[0]

    rows = rank_similarities(audio_emb, text_emb, texts)
    table = [[r["rank"], r["text"], round(r["cosine"], 4), round(r["diff_from_mean"], 4)] for r in rows]

    info = f"音声長: {len(wav) / embedder.sample_rate:.2f} 秒 / テキスト候補: {len(texts)} 件"
    if len(texts) >= 2:
        tt = text_emb @ text_emb.T
        off_diag = tt[~np.eye(len(texts), dtype=bool)]
        info += f" / 参考: テキスト同士の平均類似度 {off_diag.mean():.4f}"
    return table, info


def build_ui(embedder, cfg: dict):
    import gradio as gr

    prefix_names = list(cfg["text_prefixes"])

    def _run(raw_texts, prefix_name, audio):
        try:
            return compute(embedder, cfg, raw_texts, prefix_name, audio)
        except ValueError as e:
            raise gr.Error(str(e)) from e

    with gr.Blocks(title="EmbeddingGemma 2 音声⇔テキスト類似度") as demo:
        gr.Markdown(
            "## EmbeddingGemma 2 音声⇔テキスト類似度\n"
            "テキストを 1 行 1 候補で入力し、マイクで話して「計算」を押すと、"
            "音声埋め込みと各テキスト埋め込みのコサイン類似度を高い順に表示します。\n\n"
            "音声とテキストの埋め込みにはモダリティギャップがあり絶対値は偏りやすいので、"
            "**順位**と**候補平均との差（diff_from_mean）**で見るのがおすすめです。"
        )
        with gr.Row():
            with gr.Column():
                texts = gr.Textbox(
                    label="テキスト候補（1 行 1 候補）",
                    lines=8,
                    value="\n".join(cfg.get("default_texts", [])),
                )
                prefix = gr.Dropdown(prefix_names, value=prefix_names[0], label="テキストのプレフィックス")
            with gr.Column():
                audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="音声（マイク or ファイル）")
                button = gr.Button("計算", variant="primary")
        info = gr.Markdown()
        table = gr.Dataframe(headers=COLUMNS, label="類似度（降順）", interactive=False)
        button.click(_run, inputs=[texts, prefix, audio], outputs=[table, info])
    return demo
