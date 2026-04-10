from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from group_appeal_detector import GroupAppealDetector


@pytest.fixture
def detector():
    with patch("group_appeal_detector.GroupMentionDetector") as mock_gmd_cls, \
         patch("group_appeal_detector.StanceClassifier") as mock_sc_cls:
        mock_gmd_cls.return_value = MagicMock()
        mock_sc_cls.return_value = MagicMock()
        yield GroupAppealDetector(device="cpu")


# --- detect_mentions() ---

def test_detect_mentions_returns_span_start_end(detector):
    detector._mention_detector.detect.return_value = [
        {"word": "women", "start": 12, "end": 17, "score": 0.99},
    ]
    result = detector.detect_mentions("Support for women.")
    assert result == [{"span": "women", "start": 12, "end": 17}]


def test_detect_mentions_strips_extra_fields(detector):
    detector._mention_detector.detect.return_value = [
        {"word": "farmers", "start": 0, "end": 7, "entity_group": "GROUP", "score": 0.97},
    ]
    result = detector.detect_mentions("farmers are important.")
    assert all("score" not in m and "entity_group" not in m for m in result)


def test_detect_mentions_no_mentions(detector):
    detector._mention_detector.detect.return_value = []
    assert detector.detect_mentions("The sky is blue.") == []


def test_detect_mentions_batch_returns_nested_list(detector):
    detector._mention_detector.detect_batch.return_value = [
        [{"word": "workers", "start": 4, "end": 11, "score": 0.98}],
        [],
    ]
    result = detector.detect_mentions_batch(["The workers protested.", "No group here."])
    assert result == [[{"span": "workers", "start": 4, "end": 11}], []]


def test_detect_mentions_batch_as_df(detector):
    detector._mention_detector.detect_batch.return_value = [
        [{"word": "students", "start": 0, "end": 8, "score": 0.95}],
    ]
    result = detector.detect_mentions_batch(["students marched."], as_df=True)
    assert isinstance(result, pd.DataFrame)
    assert "span" in result.columns


# --- classify_stance() ---

def test_classify_stance_returns_predicted_stance_and_probs(detector):
    detector._stance_classifier.classify.return_value = (
        "positive",
        {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
    )
    result = detector.classify_stance("We support women.", "women")
    assert result == {
        "predicted_stance": "positive",
        "stance_probs": {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
    }



def test_classify_stance_batch_returns_list_of_dicts(detector):
    detector._stance_classifier.classify_batch.return_value = [
        ("positive", {"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
        ("negative", {"positive": 0.05, "negative": 0.9, "neutral": 0.05}),
    ]
    pairs = [("We support women.", "women"), ("We oppose immigrants.", "immigrants")]
    results = detector.classify_stance_batch(pairs)
    assert len(results) == 2
    assert results[0] == {
        "predicted_stance": "positive",
        "stance_probs": {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
    }


def test_classify_stance_batch_as_df(detector):
    detector._stance_classifier.classify_batch.return_value = [
        ("positive", {"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
    ]
    result = detector.classify_stance_batch([("Some text.", "workers")], as_df=True)
    assert isinstance(result, pd.DataFrame)
    assert "predicted_stance" in result.columns


# --- detect() ---

def test_detect_combines_mentions_and_stances(detector):
    detector._mention_detector.detect.return_value = [
        {"word": "women", "start": 12, "end": 17, "score": 0.99},
    ]
    detector._stance_classifier.classify.return_value = (
        "positive",
        {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
    )
    result = detector.detect("We support women.")
    assert result == [{
        "span": "women",
        "start": 12,
        "end": 17,
        "stance": "positive",
        "stance_probs": {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
    }]


def test_detect_no_mentions_returns_empty_list(detector):
    detector._mention_detector.detect.return_value = []
    assert detector.detect("The sky is blue.") == []
    detector._stance_classifier.classify.assert_not_called()


def test_detect_calls_classify_stance_for_each_mention(detector):
    detector._mention_detector.detect.return_value = [
        {"word": "women", "start": 0, "end": 5, "score": 0.99},
        {"word": "farmers", "start": 10, "end": 17, "score": 0.97},
    ]
    detector._stance_classifier.classify.return_value = (
        "neutral",
        {"positive": 0.1, "negative": 0.1, "neutral": 0.8},
    )
    result = detector.detect("Some text.")
    assert len(result) == 2
    assert detector._stance_classifier.classify.call_count == 2


def test_detect_batch_returns_results_per_text(detector):
    detector._mention_detector.detect_batch.return_value = [
        [{"word": "workers", "start": 4, "end": 11, "score": 0.98}],
        [{"word": "students", "start": 0, "end": 8, "score": 0.95}],
    ]
    detector._stance_classifier.classify_batch.return_value = [
        ("positive", {"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
        ("negative", {"positive": 0.05, "negative": 0.9, "neutral": 0.05}),
    ]
    results = detector.detect_batch(["The workers protested.", "Students opposed the policy."])
    assert len(results) == 2
    assert results[0][0]["span"] == "workers"
    assert results[1][0]["span"] == "students"


def test_detect_batch_text_with_no_mentions(detector):
    detector._mention_detector.detect_batch.return_value = [
        [{"word": "workers", "start": 4, "end": 11, "score": 0.98}],
        [],
    ]
    detector._stance_classifier.classify_batch.return_value = [
        ("positive", {"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
    ]
    results = detector.detect_batch(["The workers protested.", "No group here."])
    assert len(results[1]) == 0


def test_detect_batch_calls_classify_batch_once(detector):
    detector._mention_detector.detect_batch.return_value = [
        [{"word": "women", "start": 0, "end": 5, "score": 0.99}],
        [{"word": "farmers", "start": 0, "end": 7, "score": 0.97}],
    ]
    detector._stance_classifier.classify_batch.return_value = [
        ("positive", {"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
        ("negative", {"positive": 0.05, "negative": 0.9, "neutral": 0.05}),
    ]
    detector.detect_batch(["Text one.", "Text two."])
    detector._stance_classifier.classify_batch.assert_called_once()


def test_detect_batch_as_df(detector):
    detector._mention_detector.detect_batch.return_value = [
        [{"word": "workers", "start": 4, "end": 11, "score": 0.98}],
    ]
    detector._stance_classifier.classify_batch.return_value = [
        ("positive", {"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
    ]
    result = detector.detect_batch(["The workers protested."], as_df=True)
    assert isinstance(result, pd.DataFrame)
    assert "span" in result.columns
    assert "stance" in result.columns
