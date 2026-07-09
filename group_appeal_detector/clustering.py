import re
import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from .utils import to_dataframe
from .exceptions import InputTypeError, InputValueError, ModelLoadError
from ._validation import validate_device, validate_str_list
from typing import Any


class ModelMask(nn.Module):
    """
    Encoder model for contrastive learning that extracts representations
    from the [MASK] token position in a pretrained Transformer model.

    The model encodes input sequences, takes the hidden state at each
    sequence's single [MASK] position, projects it into a lower-dimensional
    space, and applies L2 normalization.
    """

    def __init__(
        self,
        tokenizer: Any,
        pretrained_model_name: str = "bert-base-uncased",
        proj_dim: int = 128,
    ) -> None:
        """
        Initialization of the class object.

        Args:
            tokenizer (Any): BERT-compatible tokenizer.
            pretrained_model_name (str): Model name of the BERT model to use for finetuning.
            proj_dim (int): Dimensionality of the hidden space that should be used for final projection.

        Returns:
            None
        """
        super().__init__()
        _config = AutoConfig.from_pretrained(pretrained_model_name)
        self.encoder: nn.Module = AutoModel.from_config(_config)
        self.mask_id: int = tokenizer.mask_token_id
        self.proj_dim: int = proj_dim
        self.hidden_size: int = self.encoder.config.hidden_size
        self.projector: nn.Module = nn.Sequential(nn.Linear(self.hidden_size, proj_dim))

    def encode(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Encodes input sequences and produces both raw and projected embeddings.

        Args:
            input_ids (Tensor): Token IDs of shape (batch_size, seq_len).
            attention_mask (Tensor): Attention mask of shape (batch_size, seq_len).

        Returns:
            Tuple[Tensor, Tensor]:
                - h: Raw mask-based embeddings of shape (batch_size, hidden_size)
                - z: L2-normalized projected embeddings of shape (batch_size, proj_dim)
        """
        # run tokenized text through the BERT model and the linear projection head
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # pull out the hidden state at each sequence's single [MASK] position
        batch_idx, seq_idx = (input_ids == self.mask_id).nonzero(as_tuple=True)
        h: Tensor = outputs.last_hidden_state[batch_idx, seq_idx]

        # run mask embeddings through the projector and normalize
        z: Tensor = self.projector(h)
        z: Tensor = F.normalize(z, p=2, dim=1)

        return h, z


class GroupMentionClusterer:
    _REPO_ID = "maxwlnd/cl_mention_embedding"

    def __init__(self, mentions: list[str], device: str = "cpu"):
        """Initializes the clusterer by loading the embedding model.

        Args:
            mentions: A list of social group mention strings to cluster.
            device: The device to run inference on. Either ``cpu``, ``cuda``,
                or ``mps``.

        Raises:
            InputTypeError: If ``mentions`` is not a list of strings, or
                ``device`` is not a string.
            InputValueError: If ``mentions`` is empty, or ``device`` is not
                a supported device type.
            ModelLoadError: If the tokenizer, model, or checkpoint fails to load.
        """
        validate_str_list(mentions, "mentions")
        if not mentions:
            raise InputValueError("mentions must not be empty.")
        validate_device(device)

        self.mentions = mentions
        self.device = torch.device(device)

        # construct the model class and load all fine-tuned weights
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self._REPO_ID)
            self.model = ModelMask(tokenizer=self.tokenizer)
            checkpoint_path = hf_hub_download(self._REPO_ID, "model.safetensors")
            self.model.load_state_dict(load_file(checkpoint_path))
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load group mention clustering model '{self._REPO_ID}': {e}"
            ) from e

        self.model.to(self.device)
        self.model.eval()

    def embed(self, max_len: int = 32, batch_size: int = 32) -> Tensor:
        """Computes and caches L2-normalized embeddings for all mentions.

        Args:
            max_len: Maximum token length for the input template.
            batch_size: Number of mentions encoded at once.

        Returns:
            A tensor of shape ``(n_mentions, 128)`` containing the embeddings.
        """
        # just load the embeddings if they have already been computed
        if hasattr(self, "_embeddings"):
            return self._embeddings

        all_embeddings = []
        mask_token = self.tokenizer.mask_token

        with torch.no_grad():
            for start in range(0, len(self.mentions), batch_size):
                # run all mentions through the template and tokenize them
                batch_texts = [
                    f"Social group of {m} is: {mask_token}."
                    for m in self.mentions[start : start + batch_size]
                ]
                enc = self.tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                # run the input tokens through the model and append the normalized embedding
                _, z = self.model.encode(input_ids, attention_mask)
                all_embeddings.append(z.cpu())

        self._embeddings = torch.cat(all_embeddings, dim=0)
        return self._embeddings

    def find_optimal_k(
        self,
        k_range: tuple[int, int] = (2, 30),
        metric: str = "silhouette",
        dictionary_df: pd.DataFrame | None = None,
        visualize: bool = True,
    ) -> tuple[int, list[float]]:
        """Finds the optimal number of clusters by maximizing a validation metric.

        Args:
            k_range: The range of k values to evaluate, inclusive.
            metric: Validation metric to use. Either ``silhouette`` (internal)
                or ``nmi`` (external, requires ``dictionary_df``).
            dictionary_df: A DataFrame where each column is a social group
                category and each row contains example terms. Required when
                ``metric='nmi'``.
            visualize: If ``True``, plots the metric scores across k values.

        Returns:
            A tuple of the best k and a list of scores for each evaluated k.

        Raises:
            InputTypeError: If ``k_range`` is not a tuple of two ints.
            InputValueError: If ``metric`` is not ``silhouette`` or ``nmi``;
                if ``metric='nmi'`` and ``dictionary_df`` is not provided; or
                if ``k_range`` is out of bounds (min below 2, min above max,
                or max above the number of mentions).
        """
        if metric not in ("silhouette", "nmi"):
            raise InputValueError(
                f"Unsupported metric '{metric}'. Expected 'silhouette' or 'nmi'."
            )
        if metric == "nmi" and dictionary_df is None:
            raise InputValueError("dictionary_df is required when metric='nmi'")

        # compute the embeddings
        embeddings = self.embed().numpy()

        if (
            not isinstance(k_range, tuple)
            or len(k_range) != 2
            or not all(isinstance(k, int) and not isinstance(k, bool) for k in k_range)
        ):
            raise InputTypeError("k_range must be a tuple of two ints (min_k, max_k).")
        min_k, max_k = k_range
        if min_k < 2:
            raise InputValueError(f"k_range minimum must be at least 2, got {min_k}.")
        if min_k > max_k:
            raise InputValueError(
                f"k_range minimum ({min_k}) must not exceed its maximum ({max_k})."
            )
        if max_k > embeddings.shape[0]:
            raise InputValueError(
                f"k_range maximum ({max_k}) cannot exceed the number of mentions "
                f"({embeddings.shape[0]})."
            )

        # precompute dictionary matches if NMI is requested
        if metric == "nmi":
            category_regex, group_lookup = _create_category_regex(dictionary_df)
            matches = [
                _match_dictionary(category_regex, group_lookup, m)
                for m in self.mentions
            ]
            dict_mask = np.array([in_dict for in_dict, _ in matches])
            dict_categories = np.array([cat for _, cat in matches])[dict_mask]

        # loop through potential ks, create k-means clustering and compute either silhouette or NMI-score
        scores = {}
        for k in range(min_k, max_k + 1):
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(
                embeddings
            )
            if metric == "silhouette":
                scores[k] = silhouette_score(embeddings, labels)
            elif metric == "nmi":
                scores[k] = normalized_mutual_info_score(
                    dict_categories, labels[dict_mask]
                )

        # take the k that produces the best score
        best_k = max(scores, key=scores.__getitem__)

        # visualize score development as a line-chart if specified
        if visualize:
            ks = list(scores.keys())
            score_values = list(scores.values())

            plt.figure(figsize=(12, 8))
            plt.plot(ks, score_values)
            plt.axvline(best_k, linestyle="--", label=f"Optimal k ({metric} score)")
            plt.grid(True, alpha=0.7)
            plt.ylim(0, 1)
            plt.xlabel("Number of clusters (k)")
            plt.title(f"{metric.capitalize()} score across k")
            plt.show()

        return best_k, list(scores.values())

    def cluster(
        self, n_clusters: int, as_df: bool = False
    ) -> list[dict] | pd.DataFrame:
        """Clusters the mentions into k groups using k-means.

        Args:
            n_clusters: The number of clusters to produce.
            as_df: If ``True``, returns a pandas DataFrame instead of a list.

        Returns:
            A list of dicts with keys ``mention``, ``cluster_id``, and
            ``distance_to_centroid`` (each point's Euclidean distance to
            its assigned cluster's centroid), or a DataFrame if
            ``as_df=True``.

        Raises:
            InputTypeError: If ``n_clusters`` is not an int.
            InputValueError: If ``n_clusters`` is not between 1 and the
                number of mentions.
        """
        embeddings = self.embed().numpy()

        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
            raise InputTypeError(
                f"Expected an int for n_clusters, got {type(n_clusters).__name__}."
            )
        if not (1 <= n_clusters <= embeddings.shape[0]):
            raise InputValueError(
                f"n_clusters must be between 1 and the number of mentions "
                f"({embeddings.shape[0]}), got {n_clusters}."
            )

        # run k-means clustering and store the cluster ids paired with the mention
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        
        # euclidean distance to every centroid, then keep only the one for the assigned cluster
        distances = kmeans.transform(embeddings)[np.arange(len(labels)), labels]
        results = [
            {"mention": m, "cluster_id": int(label), "distance_to_centroid": float(dist)}
            for m, label, dist in zip(self.mentions, labels, distances)
        ]
        return to_dataframe(results) if as_df else results


    def visualize_clusters_boxplot(
            self,
            cluster_df: pd.DataFrame,
            clusters: list[int] | None = None,
            cluster_col: str = 'cluster_id',
            distance_col: str = 'distance_to_centroid',
            mention_col: str = 'mention'
    ) -> None:
        """Plots a Plotly boxplot of each cluster's distances to its centroid.

        Args:
            cluster_df: DataFrame containing cluster, distance, and mention
                columns, e.g. the output of ``cluster(as_df=True)``.
            clusters: Cluster ids to include. Defaults to all unique values
                found in ``cluster_col``, sorted.
            cluster_col: Name of the column holding cluster ids.
            distance_col: Name of the column holding each point's distance
                to its cluster centroid.
            mention_col: Name of the column holding the mention text, shown
                on hover.

        Returns:
            None
        """
        if clusters is None:
            clusters = sorted(cluster_df[cluster_col].unique())

        fig = go.Figure(layout=dict(width=800, height=600))

        for cid in clusters:
            cluster_data = cluster_df[cluster_df[cluster_col] == cid]
            y_values = cluster_data[distance_col].values
            hover_texts = cluster_data[mention_col].values

            fig.add_trace(go.Box(
                y=y_values,
                name=f'Cluster {cid}',
                boxpoints='all',
                jitter=0.3,
                whiskerwidth=0.2,
                marker=dict(size=3),
                line=dict(width=1),
                text=hover_texts,
                hovertemplate='%{text}<br>%{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text="Clustered Social Group Mentions with Euclidean Distance to Centroid",
                x=0.5
            ),
            yaxis=dict(
                autorange=True,
                showgrid=True,
                zeroline=True,
                gridcolor='rgb(255, 255, 255)',
                gridwidth=1,
                zerolinecolor='rgb(255, 255, 255)',
                zerolinewidth=2,
                title=f"Euclidean distance to centroid"
            ),
            paper_bgcolor='rgb(243, 243, 243)',
            plot_bgcolor='rgb(243, 243, 243)',
            margin=dict(l=40, r=30, t=80, b=100),
            showlegend=False
        )
        
        fig.show()


def _normalize_group_name(name: str) -> str:
    """
    Normalizes the group name of a social group category to be able to be used by a combined regular expression pattern.

    Args:
        name: The social group name as provided in the dictionary column.

    Returns:
        The new social group category name as a string.
    """
    name = name.strip()
    name = re.sub(r"\W+", "_", name)
    if name[0].isdigit():
        name = f"cat_{name}"
    return name


def _create_category_regex(dictionary_df: pd.DataFrame) -> re.Pattern:
    """
    Creates a combined regular expression pattern from a dictionary DataFrame.
    Stores additionally the group name of each dictionary entry.

    Each cell in the DataFrame is treated as a pattern. Patterns may contain
    asterisks (*) to indicate wildcards:
    - '*' as a standalone token matches any single word
    - '*word' matches any prefix ending in 'word'
    - 'word*' matches any suffix starting with 'word'

    Tokens within a pattern are matched with flexible whitespace, and all
    patterns are combined using a logical OR.

    Args:
        dictionary_df (pd.DataFrame): DataFrame containing dictionary patterns.

    Returns:
        re.Pattern: A combined regular expression pattern matching all dictionary entries.
    """
    # empty list to store all individual patterns in
    category_patterns: list[str] = []

    # dictionary to map safe group name to original category name
    group_to_category: dict[str, str] = {}

    for category in dictionary_df.columns:
        # normalize group name and append to dictionary
        safe_name: str = _normalize_group_name(category)
        group_to_category[safe_name] = category
        # get all valid patterns
        patterns = dictionary_df[category].dropna()
        # empty list to store patterns for this category
        regex_patterns = []

        # loop over all patterns within the category
        for pat in patterns:
            # split into indvidual words/tokens
            tokens: str = pat.split()
            regex_parts: list[str] = []

            # loop over all tokens and append regular expression to list
            for token in tokens:
                if token == "*":
                    regex_parts.append(r"\w+")
                elif token.startswith("*") and len(token) > 1:
                    word = token[1:]
                    regex_parts.append(rf"\w*{re.escape(word)}")
                elif token.endswith("*") and len(token) > 1:
                    word = token[:-1]
                    regex_parts.append(rf"{re.escape(word)}\w*")
                else:
                    regex_parts.append(re.escape(token))

            # combine all regex parts for the single pattern and append
            regex_patterns.append(r"\b" + r"\s+".join(regex_parts) + r"\b")

        # append all category patterns to the general list
        if regex_patterns:
            category_patterns.append(f"(?P<{safe_name}>{'|'.join(regex_patterns)})")

    # combine all patterns across categories to a single regex
    combined: str = "|".join(category_patterns)

    return re.compile(combined, flags=re.IGNORECASE), group_to_category


def _match_dictionary(
    category_regex: str, group_lookup: dict[str, str], text: str
) -> tuple[bool, str]:
    """
    Applies regular expression to a text string.
    Returns if it found a match and if so the corresponding social group.

    Args:
        category_regex (str): Regular expression with social group search terms
        group_lookup (Dict[str, str]): Dictionary mapping safe social group names to the original ones
        text (str): The text string to search in

    Returns:
        Tuple[bool, str]:
            - Boolean indicating if there is a match
            - Actual social group name if there is a match, None otherwise
    """
    m = category_regex.search(text)
    if not m:
        return False, None
    return True, group_lookup[m.lastgroup]
