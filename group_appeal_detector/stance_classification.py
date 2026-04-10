import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class StanceClassifier:
    _STANCES = ["positive", "negative", "neutral"]

    def __init__(self, device: str = "cpu"):
        model_id = "maxwlnd/socialgroup_stance_classification_nli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.device = torch.device(device)
        self.model.to(self.device)

    def classify(self, text: str, target_group: str) -> tuple[str, dict[str, float]]:
        # raise TypeError if either text or target group are not a string
        if not isinstance(text, str):
            raise TypeError(f"Expected a string for text, got {type(text).__name__}.")
        if not isinstance(target_group, str):
            raise TypeError(f"Expected a string for target_group, got {type(target_group).__name__}.")
        
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
    
    def classify_batch(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[tuple[str, dict[str, float]]]:
        """
        Classify stance for a list of (text, target_group) pairs.
        Each pair produces 3 NLI inputs, so effective batch size is batch_size * 3.
        """
        # loop over all pairs inside the batch
        results = []
        for i in range(0, len(pairs), batch_size):
            # construct the batch manually
            batch = pairs[i:i + batch_size]
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
                probs = entail_probs[j * 3:(j + 1) * 3]
                stance_probs = dict(zip(self._STANCES, probs))
                predicted_stance = max(stance_probs, key=stance_probs.__getitem__)
                results.append((predicted_stance, stance_probs))
                
        return results