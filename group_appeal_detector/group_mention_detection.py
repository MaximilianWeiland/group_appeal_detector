from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from .exceptions import ModelLoadError
from ._validation import (
    validate_device,
    validate_positive_int,
    validate_str,
    validate_str_list,
)


class GroupMentionDetector:
    _TOKENIZER_ID = "roberta-base"
    _MODEL_ID = "maxwlnd/roberta_group_mention_detector"

    def __init__(self, device: str = "cpu"):
        """Initialises the detector by loading the token-classification model.

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
            tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_ID)
            model = AutoModelForTokenClassification.from_pretrained(self._MODEL_ID)
            self._pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device,
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load group mention detection model '{self._MODEL_ID}': {e}"
            ) from e

    def detect(self, text: str) -> list[dict]:
        validate_str(text, "text")
        return self._pipeline(text)

    def detect_batch(self, texts: list[str], batch_size: int = 32) -> list[list[dict]]:
        validate_str_list(texts, "texts")
        validate_positive_int(batch_size, "batch_size")
        return self._pipeline(texts, batch_size=batch_size)
