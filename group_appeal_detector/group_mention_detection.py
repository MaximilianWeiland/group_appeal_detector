from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class GroupMentionDetector:
    def __init__(self, device: str = "cpu"):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForTokenClassification.from_pretrained(
            "maxwlnd/roberta_group_mention_detector"
        )
        self._pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )

    def detect(self, text: str) -> list[dict]:
        return self._pipeline(text)

    def detect_batch(self, texts: list[str], batch_size: int = 32) -> list[list[dict]]:
        return self._pipeline(texts, batch_size=batch_size)
