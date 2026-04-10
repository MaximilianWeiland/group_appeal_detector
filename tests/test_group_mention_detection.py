from unittest.mock import MagicMock, patch
import pytest
from group_appeal_detector.group_mention_detection import GroupMentionDetector


PIPELINE_OUTPUT_SINGLE = [
    {"word": "women", "start": 12, "end": 17, "entity_group": "GROUP", "score": 0.99},
    {"word": "farmers", "start": 22, "end": 29, "entity_group": "GROUP", "score": 0.97},
]

PIPELINE_OUTPUT_BATCH = [
    [{"word": "workers", "start": 4, "end": 11, "entity_group": "GROUP", "score": 0.98}],
    [{"word": "students", "start": 0, "end": 8, "entity_group": "GROUP", "score": 0.95}],
]


@pytest.fixture
def detector():
    with patch("group_appeal_detector.group_mention_detection.AutoTokenizer") as mock_tok, \
         patch("group_appeal_detector.group_mention_detection.AutoModelForTokenClassification") as mock_model, \
         patch("group_appeal_detector.group_mention_detection.pipeline") as mock_pipeline:
        mock_pipe_instance = MagicMock()
        mock_pipeline.return_value = mock_pipe_instance
        d = GroupMentionDetector(device="cpu")
        d._pipeline = mock_pipe_instance
        yield d


def test_detect_returns_pipeline_output(detector):
    detector._pipeline.return_value = PIPELINE_OUTPUT_SINGLE
    result = detector.detect("Support for women and farmers.")
    assert result == PIPELINE_OUTPUT_SINGLE


def test_detect_calls_pipeline_with_text(detector):
    detector._pipeline.return_value = []
    detector.detect("Some text.")
    detector._pipeline.assert_called_once_with("Some text.")


def test_detect_no_mentions(detector):
    detector._pipeline.return_value = []
    result = detector.detect("The sky is blue.")
    assert result == []


def test_detect_batch_returns_pipeline_output(detector):
    texts = ["The workers protested.", "Students marched downtown."]
    detector._pipeline.return_value = PIPELINE_OUTPUT_BATCH
    result = detector.detect_batch(texts)
    assert result == PIPELINE_OUTPUT_BATCH


def test_detect_batch_passes_batch_size(detector):
    texts = ["Text one.", "Text two."]
    detector._pipeline.return_value = [[], []]
    detector.detect_batch(texts, batch_size=8)
    detector._pipeline.assert_called_once_with(texts, batch_size=8)


def test_detect_batch_default_batch_size(detector):
    texts = ["Text one."]
    detector._pipeline.return_value = [[]]
    detector.detect_batch(texts)
    detector._pipeline.assert_called_once_with(texts, batch_size=32)


def test_detect_batch_empty_input(detector):
    detector._pipeline.return_value = []
    result = detector.detect_batch([])
    assert result == []


def test_detect_raises_on_non_string_input(detector):
    with pytest.raises(TypeError):
        detector.detect(["some", "list"])
