from unittest.mock import MagicMock, patch
import pytest
import torch
from group_appeal_detector.stance_classification import StanceClassifier


def make_logits(positive_prob, negative_prob, neutral_prob):
    """
    Build a (3, 3) logit tensor such that after softmax along dim=-1,
    column 0 (entailment) of each row approximates the desired probability.
    A large positive value in column 0 drives its softmax entry toward 1.
    """

    def to_logit(p):
        return 10.0 if p > 0.5 else -10.0

    return torch.tensor(
        [
            [to_logit(positive_prob), 0.0],
            [to_logit(negative_prob), 0.0],
            [to_logit(neutral_prob), 0.0],
        ]
    )


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


def setup_model_output(classifier, logits: torch.Tensor):
    mock_outputs = MagicMock()
    mock_outputs.logits = logits
    classifier.model.return_value = mock_outputs
    classifier.tokenizer.return_value.to.return_value = {}


# --- classify() ---


def test_classify_returns_predicted_stance_and_probs(classifier):
    setup_model_output(
        classifier,
        make_logits(positive_prob=0.9, negative_prob=0.05, neutral_prob=0.05),
    )
    stance, probs = classifier.classify("We support women.", "women")
    assert stance == "positive"
    assert set(probs.keys()) == {"positive", "negative", "neutral"}


def test_classify_selects_highest_probability_stance(classifier):
    setup_model_output(
        classifier,
        make_logits(positive_prob=0.05, negative_prob=0.9, neutral_prob=0.05),
    )
    stance, _ = classifier.classify("We oppose immigrants.", "immigrants")
    assert stance == "negative"


def test_classify_constructs_correct_hypotheses(classifier):
    setup_model_output(
        classifier,
        make_logits(positive_prob=0.9, negative_prob=0.05, neutral_prob=0.05),
    )
    classifier.classify("Some text.", "farmers")
    classifier.tokenizer.assert_called_once_with(
        ["Some text.", "Some text.", "Some text."],
        [
            "The text is positive towards farmers.",
            "The text is negative towards farmers.",
            "The text is neutral, or contains no stance, towards farmers.",
        ],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )


def test_classify_probs_sum_to_one(classifier):
    setup_model_output(
        classifier, make_logits(positive_prob=0.6, negative_prob=0.3, neutral_prob=0.1)
    )
    _, probs = classifier.classify("Some text.", "workers")
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_classify_raises_on_non_string_text(classifier):
    with pytest.raises(TypeError):
        classifier.classify(["some", "list"], "workers")


def test_classify_raises_on_non_string_target_group(classifier):
    with pytest.raises(TypeError):
        classifier.classify("Some text.", ["workers"])


# --- classify_batch() ---


def test_classify_batch_empty_input(classifier):
    assert classifier.classify_batch([]) == []


def test_classify_batch_returns_one_result_per_pair(classifier):
    logits = make_logits(
        positive_prob=0.9, negative_prob=0.05, neutral_prob=0.05
    ).repeat(2, 1)
    setup_model_output(classifier, logits)
    pairs = [("Text one.", "group A"), ("Text two.", "group B")]
    results = classifier.classify_batch(pairs)
    assert len(results) == 2


def test_classify_batch_result_structure(classifier):
    setup_model_output(
        classifier,
        make_logits(positive_prob=0.9, negative_prob=0.05, neutral_prob=0.05),
    )
    results = classifier.classify_batch([("Some text.", "workers")])
    stance, probs = results[0]
    assert stance in {"positive", "negative", "neutral"}
    assert set(probs.keys()) == {"positive", "negative", "neutral"}


def test_classify_batch_respects_batch_size(classifier):
    setup_model_output(
        classifier,
        make_logits(positive_prob=0.9, negative_prob=0.05, neutral_prob=0.05),
    )
    pairs = [("Text one.", "group A"), ("Text two.", "group B")]
    classifier.classify_batch(pairs, batch_size=1)
    assert classifier.tokenizer.call_count == 2
