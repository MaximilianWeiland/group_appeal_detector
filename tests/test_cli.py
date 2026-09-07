import json
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import group_appeal_detector.cli as cli
from group_appeal_detector.exceptions import InputValueError


@pytest.fixture
def mock_detector():
    with patch("group_appeal_detector.cli.GroupAppealDetector") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_clusterer():
    with patch("group_appeal_detector.cli.GroupMentionClusterer") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield mock_cls, instance


def run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["group-appeal-detector"] + argv)
    return cli.main()


# --- top-level parser ---


def test_no_command_raises_systemexit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["group-appeal-detector"])
    with pytest.raises(SystemExit):
        cli.main()


# --- detect ---


def test_detect_single_text(mock_detector, monkeypatch, capsys):
    mock_detector.detect.return_value = [
        {"span": "women", "start": 0, "end": 5, "stance": "positive", "stance_probs": {}}
    ]
    rc = run_cli(monkeypatch, ["detect", "We support women."])
    assert rc == 0
    mock_detector.detect.assert_called_once_with("We support women.")
    out = json.loads(capsys.readouterr().out)
    assert out[0]["span"] == "women"


def test_detect_batch_from_file(tmp_path, mock_detector, monkeypatch):
    file = tmp_path / "texts.txt"
    file.write_text("Text one.\nText two.\n")
    mock_detector.detect_batch.return_value = [[], []]
    rc = run_cli(monkeypatch, ["detect", "--file", str(file)])
    assert rc == 0
    mock_detector.detect_batch.assert_called_once_with(
        ["Text one.", "Text two."], batch_size=32, as_df=False
    )


def test_detect_batch_csv_format(tmp_path, mock_detector, monkeypatch, capsys):
    file = tmp_path / "texts.txt"
    file.write_text("Text one.\n")
    df = pd.DataFrame([{"text_idx": 0, "span": "women", "stance": "positive"}])
    mock_detector.detect_batch.return_value = df
    rc = run_cli(monkeypatch, ["detect", "--file", str(file), "--format", "csv"])
    assert rc == 0
    mock_detector.detect_batch.assert_called_once_with(
        ["Text one."], batch_size=32, as_df=True
    )
    out = capsys.readouterr().out
    assert "span" in out
    assert "women" in out


def test_detect_missing_input_raises(mock_detector, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["group-appeal-detector", "detect"])
    with pytest.raises(SystemExit):
        cli.main()


# --- detect-mentions ---


def test_detect_mentions_single_text(mock_detector, monkeypatch, capsys):
    mock_detector.detect_mentions.return_value = [
        {"span": "farmers", "start": 0, "end": 7}
    ]
    rc = run_cli(monkeypatch, ["detect-mentions", "farmers are important."])
    assert rc == 0
    mock_detector.detect_mentions.assert_called_once_with("farmers are important.")
    out = json.loads(capsys.readouterr().out)
    assert out == [{"span": "farmers", "start": 0, "end": 7}]


def test_detect_mentions_missing_input_raises(mock_detector, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["group-appeal-detector", "detect-mentions"])
    with pytest.raises(SystemExit):
        cli.main()


# --- classify-stance ---


def test_classify_stance_single(mock_detector, monkeypatch, capsys):
    mock_detector.classify_stance.return_value = {
        "predicted_stance": "positive",
        "stance_probs": {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
    }
    rc = run_cli(monkeypatch, ["classify-stance", "We support women.", "women"])
    assert rc == 0
    mock_detector.classify_stance.assert_called_once_with("We support women.", "women")
    out = json.loads(capsys.readouterr().out)
    assert out["predicted_stance"] == "positive"


def test_classify_stance_from_csv_file(tmp_path, mock_detector, monkeypatch):
    file = tmp_path / "pairs.csv"
    file.write_text(
        "text,target_group\n"
        "We support women.,women\n"
        "We oppose immigrants.,immigrants\n"
    )
    mock_detector.classify_stance_batch.return_value = [
        {"predicted_stance": "positive", "stance_probs": {}},
        {"predicted_stance": "negative", "stance_probs": {}},
    ]
    rc = run_cli(monkeypatch, ["classify-stance", "--file", str(file)])
    assert rc == 0
    pairs = mock_detector.classify_stance_batch.call_args[0][0]
    assert pairs == [
        ("We support women.", "women"),
        ("We oppose immigrants.", "immigrants"),
    ]


def test_classify_stance_missing_target_group_raises(mock_detector, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["group-appeal-detector", "classify-stance", "only text"]
    )
    with pytest.raises(SystemExit):
        cli.main()


# --- cluster ---


def test_cluster_with_positional_mentions(mock_clusterer, monkeypatch, capsys):
    mock_cls, instance = mock_clusterer
    instance.cluster.return_value = [
        {"mention": "women", "cluster_id": 0, "distance_to_centroid": 0.1}
    ]
    rc = run_cli(monkeypatch, ["cluster", "women", "farmers", "--n-clusters", "2"])
    assert rc == 0
    mock_cls.assert_called_once_with(["women", "farmers"], device="cpu")
    instance.cluster.assert_called_once_with(2, as_df=False)


def test_cluster_from_file(tmp_path, mock_clusterer, monkeypatch):
    file = tmp_path / "mentions.txt"
    file.write_text("women\nfarmers\n")
    mock_cls, instance = mock_clusterer
    instance.cluster.return_value = []
    rc = run_cli(monkeypatch, ["cluster", "--file", str(file), "--n-clusters", "2"])
    assert rc == 0
    mock_cls.assert_called_once_with(["women", "farmers"], device="cpu")


def test_cluster_missing_mentions_raises(mock_clusterer, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["group-appeal-detector", "cluster", "--n-clusters", "2"]
    )
    with pytest.raises(SystemExit):
        cli.main()


def test_cluster_output_to_csv_file(tmp_path, mock_clusterer, monkeypatch):
    mock_cls, instance = mock_clusterer
    instance.cluster.return_value = pd.DataFrame(
        [{"mention": "women", "cluster_id": 0, "distance_to_centroid": 0.1}]
    )
    out_file = tmp_path / "out.csv"
    rc = run_cli(
        monkeypatch,
        ["cluster", "women", "--n-clusters", "1", "--format", "csv", "-o", str(out_file)],
    )
    assert rc == 0
    assert "women" in out_file.read_text()


# --- find-optimal-k ---


def test_find_optimal_k(mock_clusterer, monkeypatch, capsys):
    mock_cls, instance = mock_clusterer
    instance.find_optimal_k.return_value = (3, [0.1, 0.2, 0.3])
    rc = run_cli(
        monkeypatch,
        ["find-optimal-k", "women", "farmers", "workers", "--k-min", "2", "--k-max", "4"],
    )
    assert rc == 0
    instance.find_optimal_k.assert_called_once_with(
        k_range=(2, 4), metric="silhouette", dictionary_df=None, visualize=False
    )
    out = json.loads(capsys.readouterr().out)
    assert out == {"best_k": 3, "scores": [0.1, 0.2, 0.3]}


def test_find_optimal_k_with_dictionary(tmp_path, mock_clusterer, monkeypatch):
    mock_cls, instance = mock_clusterer
    instance.find_optimal_k.return_value = (2, [0.5, 0.6])
    dict_file = tmp_path / "dictionary.csv"
    dict_file.write_text("women,men\nwoman,man\n")
    rc = run_cli(
        monkeypatch,
        [
            "find-optimal-k",
            "women",
            "men",
            "--metric",
            "nmi",
            "--dictionary",
            str(dict_file),
        ],
    )
    assert rc == 0
    call_kwargs = instance.find_optimal_k.call_args.kwargs
    assert call_kwargs["metric"] == "nmi"
    assert isinstance(call_kwargs["dictionary_df"], pd.DataFrame)


def test_find_optimal_k_missing_mentions_raises(mock_clusterer, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["group-appeal-detector", "find-optimal-k"])
    with pytest.raises(SystemExit):
        cli.main()


# --- error handling ---


def test_main_catches_group_appeal_detector_error(mock_detector, monkeypatch, capsys):
    mock_detector.detect.side_effect = InputValueError("bad input")
    rc = run_cli(monkeypatch, ["detect", "some text"])
    assert rc == 1
    assert "bad input" in capsys.readouterr().err
