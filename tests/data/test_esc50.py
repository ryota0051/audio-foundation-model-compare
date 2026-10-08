from egaudio.data.esc50 import label_text, load_meta


def test_label_text():
    assert label_text("chirping_birds") == "chirping birds"


def test_load_meta_filters_folds(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text(
        "filename,fold,target,category,esc10,src_file,take\n"
        "a.wav,1,0,dog,True,1,A\nb.wav,2,1,rooster,False,2,A\n",
        encoding="utf-8",
    )
    df = load_meta(csv, audio_dir=tmp_path, folds=[2])
    assert list(df["category"]) == ["rooster"]
    assert df["path"].iloc[0].endswith("b.wav")
    assert len(load_meta(csv, audio_dir=tmp_path, folds=None)) == 2
