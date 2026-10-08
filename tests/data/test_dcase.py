import pytest

from egaudio.data.dcase import list_clips, parse_filename


def test_parse_train_target():
    m = parse_filename("section_00_target_train_normal_0001_spd_2.wav")
    assert m == {"section": "00", "domain": "target", "split": "train", "label": "normal", "is_anomaly": 0}


def test_parse_test_anomaly_noattribute():
    m = parse_filename("section_00_source_test_anomaly_0000_noAttribute.wav")
    assert m["domain"] == "source" and m["split"] == "test" and m["is_anomaly"] == 1


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_filename("foo.wav")


def test_list_clips(tmp_path):
    d = tmp_path / "fan" / "train"
    d.mkdir(parents=True)
    for n in ["section_00_source_train_normal_0000_a.wav", "section_00_target_train_normal_0000_a.wav"]:
        (d / n).write_bytes(b"")
    (d / "readme.txt").write_text("x")
    df = list_clips(tmp_path, "fan", "train")
    assert len(df) == 2
    assert sorted(df["domain"]) == ["source", "target"]
    assert set(df.columns) >= {"path", "filename", "domain", "is_anomaly"}
