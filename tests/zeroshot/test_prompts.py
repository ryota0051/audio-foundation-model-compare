import pytest

from egaudio.zeroshot.prompts import build_texts


def test_build_texts_templates_and_prefix():
    classes = ["dog", "chirping_birds"]
    natural = {"dog": "a dog barking", "chirping_birds": "birds chirping"}
    assert build_texts(classes, "{label}", "{text}", natural) == ["dog", "chirping birds"]
    assert build_texts(classes, "the sound of {natural}", "task: classification | query: {text}", natural) == [
        "task: classification | query: the sound of a dog barking",
        "task: classification | query: the sound of birds chirping",
    ]


def test_build_texts_missing_natural_raises():
    with pytest.raises(KeyError):
        build_texts(["cat"], "{natural}", "{text}", {})
