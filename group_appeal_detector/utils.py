import pandas as pd


def to_dataframe(results: list[dict] | list[list[dict]], text_idx: bool = True) -> pd.DataFrame:
    if results and isinstance(results[0], list):
        rows = []
        for i, group in enumerate(results):
            for item in group:
                row = {"text_idx": i, **item} if text_idx else item
                rows.append(row)
    else:
        rows = results

    df = pd.DataFrame(rows)
    if "stance_probs" in df.columns:
        probs_df = pd.json_normalize(df.pop("stance_probs")).add_prefix("stance_prob_")
        df = pd.concat([df.reset_index(drop=True), probs_df], axis=1)
    return df
