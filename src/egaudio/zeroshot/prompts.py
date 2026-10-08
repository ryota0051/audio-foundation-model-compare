"""クラス名からテキストクエリを組み立てる。"""

from egaudio.data.esc50 import label_text


def build_texts(classes: list[str], template: str, prefix: str, natural: dict[str, str]) -> list[str]:
    texts = []
    for c in classes:
        fields = {"label": label_text(c)}
        if "{natural}" in template:
            fields["natural"] = natural[c]
        texts.append(prefix.format(text=template.format(**fields)))
    return texts
