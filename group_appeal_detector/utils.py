import pandas as pd


def to_dataframe(
    results: list[dict] | list[list[dict]], text_idx: bool = True
) -> pd.DataFrame:
    """Converts detection results to a pandas DataFrame.

    Accepts either a flat list of dicts or a nested list (one list per text).
    If ``stance_probs`` is present, it is expanded into separate columns
    prefixed with ``stance_prob_``.

    Args:
        results: A flat list of dicts or a nested list of dicts.
        text_idx: If ``True``, adds a ``text_idx`` column indicating which
            input text each row belongs to. Only applied for nested lists.

    Returns:
        A pandas DataFrame with one row per detection result.
    """
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
