import warnings
from unittest.mock import MagicMock, patch

from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import pytest
import torch
from group_appeal_detector.clustering import (
    GroupMentionClusterer,
    ModelMask,
    _normalize_group_name,
    _create_category_regex,
    _match_dictionary,
)


# --- _normalize_group_name ---


def test_normalize_strips_whitespace():
    assert _normalize_group_name("  women  ") == "women"


def test_normalize_replaces_non_word_chars():
    assert _normalize_group_name("working-class people") == "working_class_people"


def test_normalize_prepends_cat_if_starts_with_digit():
    assert _normalize_group_name("18-24 year olds").startswith("cat_")


# --- _create_category_regex / _match_dictionary ---


@pytest.fixture
def simple_dict_df():
    return pd.DataFrame(
        {"workers": ["workers", "laborers"], "students": ["students", None]}
    )


def test_match_dictionary_exact_match(simple_dict_df):
    regex, lookup = _create_category_regex(simple_dict_df)
    found, category = _match_dictionary(regex, lookup, "The workers protested.")
    assert found is True
    assert category == "workers"


def test_match_dictionary_case_insensitive(simple_dict_df):
    regex, lookup = _create_category_regex(simple_dict_df)
    found, _ = _match_dictionary(regex, lookup, "The WORKERS protested.")
    assert found is True


def test_match_dictionary_no_match(simple_dict_df):
    regex, lookup = _create_category_regex(simple_dict_df)
    found, category = _match_dictionary(regex, lookup, "The sky is blue.")
    assert found is False
    assert category is None


def test_match_dictionary_wildcard_standalone():
    df = pd.DataFrame({"group": ["* rights"]})
    regex, lookup = _create_category_regex(df)
    found, _ = _match_dictionary(regex, lookup, "women rights matter.")
    assert found is True


def test_match_dictionary_wildcard_suffix():
    df = pd.DataFrame({"group": ["work*"]})
    regex, lookup = _create_category_regex(df)
    found, _ = _match_dictionary(regex, lookup, "The workers are striking.")
    assert found is True


def test_match_dictionary_wildcard_prefix():
    df = pd.DataFrame({"group": ["*ers"]})
    regex, lookup = _create_category_regex(df)
    found, _ = _match_dictionary(regex, lookup, "The farmers protested.")
    assert found is True


# --- ModelMask._extract_mask_embedding ---


@pytest.fixture
def model_mask():
    model = ModelMask.__new__(ModelMask)
    model.mask_id = 103
    return model


def test_extract_mask_embedding_averages_mask_positions(model_mask):
    # Two mask tokens at positions 1 and 2
    input_ids = torch.tensor([[0, 103, 103, 0]])
    hidden = torch.zeros(1, 4, 8)
    hidden[0, 1] = torch.ones(8) * 2.0
    hidden[0, 2] = torch.ones(8) * 4.0
    result = model_mask._extract_mask_embedding(input_ids, hidden)
    expected = torch.ones(8) * 3.0
    assert torch.allclose(result[0], expected)


def test_extract_mask_embedding_falls_back_to_first_token(model_mask):
    # No mask tokens
    input_ids = torch.tensor([[0, 1, 2, 3]])
    hidden = torch.zeros(1, 4, 8)
    hidden[0, 0] = torch.ones(8) * 5.0
    result = model_mask._extract_mask_embedding(input_ids, hidden)
    expected = torch.ones(8) * 5.0
    assert torch.allclose(result[0], expected)


def test_extract_mask_embedding_output_shape(model_mask):
    input_ids = torch.tensor([[0, 103, 0], [103, 0, 0]])
    hidden = torch.randn(2, 3, 8)
    result = model_mask._extract_mask_embedding(input_ids, hidden)
    assert result.shape == (2, 8)


# --- GroupMentionClusterer ---


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


def setup_embed(clusterer, n_mentions: int, proj_dim: int = 4):
    embeddings = torch.randn(n_mentions, proj_dim)
    clusterer.tokenizer.return_value = {
        "input_ids": torch.zeros(n_mentions, 32, dtype=torch.long),
        "attention_mask": torch.ones(n_mentions, 32, dtype=torch.long),
    }
    clusterer.model.encode.return_value = (None, embeddings)
    return embeddings


def test_embed_builds_correct_templates(clusterer):
    setup_embed(clusterer, n_mentions=3)
    clusterer.embed()
    texts = clusterer.tokenizer.call_args[0][0]
    assert texts == [
        "Social group of women is: [MASK].",
        "Social group of farmers is: [MASK].",
        "Social group of workers is: [MASK].",
    ]


def test_embed_caches_result(clusterer):
    setup_embed(clusterer, n_mentions=3)
    first = clusterer.embed()
    second = clusterer.embed()
    assert clusterer.model.encode.call_count == 1
    assert torch.equal(first, second)


def test_cluster_returns_one_result_per_mention(clusterer):
    clusterer._embeddings = torch.randn(3, 4)
    results = clusterer.cluster(n_clusters=2)
    assert len(results) == 3


def test_cluster_result_structure(clusterer):
    clusterer._embeddings = torch.randn(3, 4)
    results = clusterer.cluster(n_clusters=2)
    for r in results:
        assert "mention" in r
        assert "cluster_id" in r
        assert r["mention"] in ["women", "farmers", "workers"]
        assert isinstance(r["cluster_id"], int)


def test_cluster_as_df(clusterer):
    clusterer._embeddings = torch.randn(3, 4)
    result = clusterer.cluster(n_clusters=2, as_df=True)
    assert isinstance(result, pd.DataFrame)
    assert "mention" in result.columns
    assert "cluster_id" in result.columns


def test_find_optimal_k_returns_best_k(clusterer):
    # Use clearly separable embeddings so silhouette reliably picks k=2
    clusterer._embeddings = torch.cat(
        [
            torch.zeros(5, 4),
            torch.ones(5, 4) * 100,
        ]
    )
    with warnings.catch_warnings(), patch("group_appeal_detector.clustering.plt"):
        warnings.simplefilter("ignore", ConvergenceWarning)
        best_k, scores = clusterer.find_optimal_k(k_range=(2, 3), visualize=True)
    assert best_k == 2
    assert len(scores) == 2


def test_find_optimal_k_nmi_raises_without_dictionary(clusterer):
    clusterer._embeddings = torch.randn(5, 4)
    with pytest.raises(ValueError, match="dictionary_df"):
        clusterer.find_optimal_k(k_range=(2, 3), metric="nmi", visualize=False)
