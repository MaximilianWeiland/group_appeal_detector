import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .exceptions import ModelLoadError
from ._validation import (
    validate_device,
    validate_pairs,
    validate_positive_int,
    validate_str,
)


class StanceClassifier:
    _STANCES = ["positive", "negative", "neutral"]
    _MODEL_ID = "maxwlnd/socialgroup_stance_classification_nli"

    def __init__(self, device: str = "cpu"):
        """Initialises the classifier by loading the NLI-based stance model.

        Args:
            device: The device to run inference on. Either ``cpu``, ``cuda``,
                or ``mps`` (optionally suffixed with an index, e.g. ``cuda:0``).

        Raises:
            InputTypeError: If ``device`` is not a string.
            InputValueError: If ``device`` is not a supported device type.
            ModelLoadError: If the tokenizer or model fails to load.
        """
        validate_device(device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self._MODEL_ID
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load stance classification model '{self._MODEL_ID}': {e}"
            ) from e
        self.device = torch.device(device)
        self.model.to(self.device)

    def classify(self, text: str, target_group: str) -> tuple[str, dict[str, float]]:
        # raise InputTypeError if either text or target group are not a string
        validate_str(text, "text")
        validate_str(target_group, "target_group")

        # construct hypotheses for each stance class
        hypotheses = [
            f"The text is positive towards {target_group}.",
            f"The text is negative towards {target_group}.",
            f"The text is neutral, or contains no stance, towards {target_group}.",
        ]

        # tokenize all hypotheses
        inputs = self.tokenizer(
            [text] * 3,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # run through the model, take softmax for each hypothesis
        with torch.no_grad():
            outputs = self.model(**inputs)
            entail_probs = torch.softmax(outputs.logits, dim=-1)[:, 0].tolist()

        # choose stance class with highest entailment probability
        stance_probs = dict(zip(self._STANCES, entail_probs))
        predicted_stance = max(stance_probs, key=stance_probs.__getitem__)
        return predicted_stance, stance_probs

    def classify_batch(
        self, pairs: list[tuple[str, str]], batch_size: int = 32
    ) -> list[tuple[str, dict[str, float]]]:
        """
        Classify stance for a list of (text, target_group) pairs.
        Each pair produces 3 NLI inputs, so effective batch size is batch_size * 3.

        Raises:
            InputTypeError: If ``pairs`` is not a list of ``(text, target_group)``
                string tuples, or ``batch_size`` is not an int.
            InputValueError: If ``batch_size`` is not positive.
        """
        validate_pairs(pairs, "pairs")
        validate_positive_int(batch_size, "batch_size")

        # loop over all pairs inside the batch
        results = []
        for i in range(0, len(pairs), batch_size):
            # construct the batch manually
            batch = pairs[i : i + batch_size]
            all_texts, all_hypotheses = [], []
            # store replicated texts and hypotheses in lists and tokenize
            for text, target_group in batch:
                all_texts += [text] * 3
                all_hypotheses += [
                    f"The text is positive towards {target_group}.",
                    f"The text is negative towards {target_group}.",
                    f"The text is neutral, or contains no stance, towards {target_group}.",
                ]

            # tokenize all list pairs
            inputs = self.tokenizer(
                all_texts,
                all_hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # run through the model and take softmax within each stance class
            with torch.no_grad():
                outputs = self.model(**inputs)
                entail_probs = torch.softmax(outputs.logits, dim=-1)[:, 0].tolist()

            # loop through all sentences inside the batch and take stance class with highest entailment prob
            for j in range(len(batch)):
                probs = entail_probs[j * 3 : (j + 1) * 3]
                stance_probs = dict(zip(self._STANCES, probs))
                predicted_stance = max(stance_probs, key=stance_probs.__getitem__)
                results.append((predicted_stance, stance_probs))

        return results
