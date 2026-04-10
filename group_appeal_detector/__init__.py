import logging
import pandas as pd

# suppress HF warning against unauthenticated request
_hf_filter = type("_HFUnauthFilter", (logging.Filter,), {
    "filter": lambda self, r: "unauthenticated" not in r.getMessage().lower()
})()
logging.getLogger("huggingface_hub").addFilter(_hf_filter)
logging.getLogger("huggingface_hub.utils._http").addFilter(_hf_filter)
from .group_mention_detection import GroupMentionDetector
from .stance_classification import StanceClassifier
from .clustering import GroupMentionClusterer
from .utils import to_dataframe


class GroupAppealDetector:
    def __init__(self, device: str = "cpu"):
        """Initialises the detector by loading all underlying models.

        Args:
            device: The device to run inference on. Either ``cpu``, ``cuda``,
                or ``mps``.
        """
        self._mention_detector = GroupMentionDetector(device=device)
        self._stance_classifier = StanceClassifier(device=device)

    def detect_mentions(self, text: str) -> list[dict]:
        """Detects social group mentions in a single text.

        Args:
            text: The input text to analyse.

        Returns:
            A list of dicts with keys ``span``, ``start``, and ``end``.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        return [
            {"span": m["word"], "start": m["start"], "end": m["end"]}
            for m in self._mention_detector.detect(text)
        ]

    def detect_mentions_batch(self, texts: list[str], batch_size: int = 32, as_df: bool = False) -> list[list[dict]] | pd.DataFrame:
        """Detects social group mentions in a batch of texts.

        Args:
            texts: A list of input texts to analyse.
            batch_size: Number of texts processed at once.
            as_df: If ``True``, returns a pandas DataFrame instead of a nested list.

        Returns:
            A nested list of dicts (one list per text) with keys ``span``,
            ``start``, and ``end``, or a DataFrame if ``as_df=True``.
        """
        results = [
            [{"span": m["word"], "start": m["start"], "end": m["end"]} for m in mentions]
            for mentions in self._mention_detector.detect_batch(texts, batch_size=batch_size)
        ]
        return to_dataframe(results) if as_df else results

    def classify_stance(self, text: str, target_group: str) -> dict:
        """Classifies the author's stance toward a specific group in a single text.

        Args:
            text: The input text to analyse.
            target_group: The social group to classify the stance toward.

        Returns:
            A dict with keys ``predicted_stance`` (one of ``positive``,
            ``negative``, ``neutral``) and ``stance_probs`` (a dict of
            per-class probabilities).

        Raises:
            TypeError: If ``text`` or ``target_group`` is not a string.
        """
        predicted_stance, stance_probs = self._stance_classifier.classify(text, target_group)
        return {"predicted_stance": predicted_stance, "stance_probs": stance_probs}

    def classify_stance_batch(self, pairs: list[tuple[str, str]], batch_size: int = 32, as_df: bool = False) -> list[dict] | pd.DataFrame:
        """Classifies the author's stance for a batch of (text, group) tuple pairs.

        Args:
            pairs: A list of ``(text, target_group)`` tuples.
            batch_size: Number of pairs processed at once.
            as_df: If ``True``, returns a pandas DataFrame instead of a list.

        Returns:
            A list of dicts with keys ``predicted_stance`` and ``stance_probs``,
            or a DataFrame if ``as_df=True``.
        """
        results = [
            {"predicted_stance": stance, "stance_probs": probs}
            for stance, probs in self._stance_classifier.classify_batch(pairs, batch_size)
        ]
        return to_dataframe(results) if as_df else results

    def detect(self, text: str) -> list[dict]:
        """Detects group mentions and classifies the stance toward each in a single pass.

        Args:
            text: The input text to analyze.

        Returns:
            A list of dicts with keys ``span``, ``start``, ``end``, ``stance``,
            and ``stance_probs``.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        results = []

        # detect all mentions first and classify stances for these
        for mention in self.detect_mentions(text):
            stance_result = self.classify_stance(text, mention["span"])
            results.append({
                "span": mention["span"],
                "start": mention["start"],
                "end": mention["end"],
                "stance": stance_result["predicted_stance"],
                "stance_probs": stance_result["stance_probs"],
            })
        return results

    def detect_batch(self, texts: list[str], batch_size: int = 32, as_df: bool = False) -> list[list[dict]] | pd.DataFrame:
        """Detects group mentions and classifies the stance toward each for a batch of texts.

        Args:
            texts: A list of input texts to analyse.
            batch_size: Number of texts processed at once.
            as_df: If ``True``, returns a pandas DataFrame instead of a nested list.

        Returns:
            A nested list of dicts (one list per text) with keys ``span``,
            ``start``, ``end``, ``stance``, and ``stance_probs``, or a
            DataFrame if ``as_df=True``.
        """
        # detect all mentions first and then classify stances toward all of them
        all_mentions = self.detect_mentions_batch(texts, batch_size=batch_size)
        pairs = [
            (text, mention["span"])
            for text, mentions in zip(texts, all_mentions)
            for mention in mentions
        ]
        stances = self._stance_classifier.classify_batch(pairs, batch_size=batch_size)

        results = []
        idx = 0

        # loop over all mentions inside all batches
        for mentions in all_mentions:
            sentence_results = []
            for mention in mentions:
                # extract the stance based on the index
                stance, stance_probs = stances[idx]
                idx += 1
                sentence_results.append({
                    "span": mention["span"],
                    "start": mention["start"],
                    "end": mention["end"],
                    "stance": stance,
                    "stance_probs": stance_probs,
                })
            results.append(sentence_results)
        return to_dataframe(results) if as_df else results
