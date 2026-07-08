from unittest.mock import MagicMock, patch

import pytest
import torch

from group_appeal_detector.exceptions import (
    GroupAppealDetectorError,
    InputTypeError,
    InputValueError,
    ModelLoadError,
)
from group_appeal_detector.group_mention_detection import GroupMentionDetector
from group_appeal_detector.stance_classification import StanceClassifier
from group_appeal_detector.clustering import GroupMentionClusterer


# --- exception hierarchy ---


def test_input_type_error_is_type_error_and_package_error():
    assert issubclass(InputTypeError, TypeError)
    assert issubclass(InputTypeError, GroupAppealDetectorError)


def test_input_value_error_is_value_error_and_package_error():
    assert issubclass(InputValueError, ValueError)
    assert issubclass(InputValueError, GroupAppealDetectorError)


def test_model_load_error_is_package_error():
    assert issubclass(ModelLoadError, GroupAppealDetectorError)
    assert not issubclass(ModelLoadError, (TypeError, ValueError))


# --- device validation ---


def test_group_mention_detector_rejects_non_string_device():
    with pytest.raises(InputTypeError):
        GroupMentionDetector(device=123)


def test_group_mention_detector_rejects_unsupported_device():
    with pytest.raises(InputValueError):
        GroupMentionDetector(device="tpu")


def test_stance_classifier_rejects_non_string_device():
    with pytest.raises(InputTypeError):
        StanceClassifier(device=123)


def test_stance_classifier_rejects_unsupported_device():
    with pytest.raises(InputValueError):
        StanceClassifier(device="tpu")


def test_clusterer_rejects_unsupported_device():
    with pytest.raises(InputValueError):
        GroupMentionClusterer(["women"], device="tpu")


def test_clusterer_accepts_indexed_cuda_device():
    with (
        patch("group_appeal_detector.clustering.AutoTokenizer") as mock_tok_cls,
        patch("group_appeal_detector.clustering.ModelMask") as mock_model_cls,
        patch("group_appeal_detector.clustering.hf_hub_download"),
        patch("group_appeal_detector.clustering.load_file"),
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token = "[MASK]"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.return_value = mock_model
        # should not raise despite the unusual (but validly-formed) device string
        GroupMentionClusterer(["women"], device="cuda:0")


# --- model load failures ---


def test_group_mention_detector_wraps_load_failure():
    with patch(
        "group_appeal_detector.group_mention_detection.AutoTokenizer"
    ) as mock_tok_cls:
        mock_tok_cls.from_pretrained.side_effect = OSError("repo not found")
        with pytest.raises(ModelLoadError):
            GroupMentionDetector(device="cpu")


def test_stance_classifier_wraps_load_failure():
    with patch(
        "group_appeal_detector.stance_classification.AutoTokenizer"
    ) as mock_tok_cls:
        mock_tok_cls.from_pretrained.side_effect = OSError("connection error")
        with pytest.raises(ModelLoadError):
            StanceClassifier(device="cpu")


def test_clusterer_wraps_load_failure():
    with patch("group_appeal_detector.clustering.AutoTokenizer") as mock_tok_cls:
        mock_tok_cls.from_pretrained.side_effect = OSError("connection error")
        with pytest.raises(ModelLoadError):
            GroupMentionClusterer(["women"], device="cpu")


# --- batch input validation ---


@pytest.fixture
def detector():
    with (
        patch("group_appeal_detector.group_mention_detection.AutoTokenizer"),
        patch(
            "group_appeal_detector.group_mention_detection.AutoModelForTokenClassification"
        ),
        patch(
            "group_appeal_detector.group_mention_detection.pipeline"
        ) as mock_pipeline,
    ):
        mock_pipeline.return_value = MagicMock()
        yield GroupMentionDetector(device="cpu")


def test_detect_batch_rejects_non_list_texts(detector):
    with pytest.raises(InputTypeError):
        detector.detect_batch("not a list")


def test_detect_batch_rejects_non_string_elements(detector):
    with pytest.raises(InputTypeError):
        detector.detect_batch(["ok", 123])


def test_detect_batch_rejects_non_positive_batch_size(detector):
    with pytest.raises(InputValueError):
        detector.detect_batch(["ok"], batch_size=0)


def test_detect_batch_rejects_non_int_batch_size(detector):
    with pytest.raises(InputTypeError):
        detector.detect_batch(["ok"], batch_size="8")


@pytest.fixture
def classifier():
    with (
        patch(
            "group_appeal_detector.stance_classification.AutoTokenizer"
        ) as mock_tok_cls,
        patch(
            "group_appeal_detector.stance_classification.AutoModelForSequenceClassification"
        ) as mock_model_cls,
    ):
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        yield StanceClassifier(device="cpu")


def test_classify_batch_rejects_non_list_pairs(classifier):
    with pytest.raises(InputTypeError):
        classifier.classify_batch("not a list")


def test_classify_batch_rejects_malformed_pair(classifier):
    with pytest.raises(InputTypeError):
        classifier.classify_batch([("only one element",)])


def test_classify_batch_rejects_non_string_pair_element(classifier):
    with pytest.raises(InputTypeError):
        classifier.classify_batch([("text", 123)])


def test_classify_batch_rejects_non_positive_batch_size(classifier):
    with pytest.raises(InputValueError):
        classifier.classify_batch([("text", "group")], batch_size=0)


# --- clustering parameter validation ---


@pytest.fixture
def clusterer():
    with (
        patch("group_appeal_detector.clustering.AutoTokenizer") as mock_tok_cls,
        patch("group_appeal_detector.clustering.ModelMask") as mock_model_cls,
        patch("group_appeal_detector.clustering.hf_hub_download") as mock_download,
        patch("group_appeal_detector.clustering.load_file"),
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token = "[MASK]"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.return_value = mock_model
        mock_download.return_value = "/fake/path"
        yield GroupMentionClusterer(["women", "farmers", "workers"], device="cpu")


def test_clusterer_rejects_empty_mentions():
    with pytest.raises(InputValueError):
        GroupMentionClusterer([], device="cpu")


def test_clusterer_rejects_non_string_mentions():
    with pytest.raises(InputTypeError):
        GroupMentionClusterer(["women", 123], device="cpu")


def test_cluster_rejects_n_clusters_above_mention_count(clusterer):
    clusterer._embeddings = torch.randn(3, 4)
    with pytest.raises(InputValueError):
        clusterer.cluster(n_clusters=10)


def test_cluster_rejects_non_int_n_clusters(clusterer):
    clusterer._embeddings = torch.randn(3, 4)
    with pytest.raises(InputTypeError):
        clusterer.cluster(n_clusters="2")


def test_find_optimal_k_rejects_unsupported_metric(clusterer):
    clusterer._embeddings = torch.randn(5, 4)
    with pytest.raises(InputValueError, match="metric"):
        clusterer.find_optimal_k(k_range=(2, 3), metric="bogus", visualize=False)


def test_find_optimal_k_rejects_k_range_below_two(clusterer):
    clusterer._embeddings = torch.randn(5, 4)
    with pytest.raises(InputValueError):
        clusterer.find_optimal_k(k_range=(1, 3), visualize=False)


def test_find_optimal_k_rejects_inverted_k_range(clusterer):
    clusterer._embeddings = torch.randn(5, 4)
    with pytest.raises(InputValueError):
        clusterer.find_optimal_k(k_range=(4, 2), visualize=False)


def test_find_optimal_k_rejects_k_range_above_mention_count(clusterer):
    clusterer._embeddings = torch.randn(5, 4)
    with pytest.raises(InputValueError):
        clusterer.find_optimal_k(k_range=(2, 6), visualize=False)
