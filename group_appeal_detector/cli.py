import argparse
import json
import sys

import pandas as pd

from . import GroupAppealDetector
from .clustering import GroupMentionClusterer
from .exceptions import GroupAppealDetectorError


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_result(result, output: str | None, fmt: str) -> None:
    if isinstance(result, pd.DataFrame):
        if fmt == "csv":
            result.to_csv(output if output else sys.stdout, index=False)
        else:
            text = result.to_json(orient="records", indent=2)
            _write_text(text, output)
    else:
        text = json.dumps(result, indent=2)
        _write_text(text, output)


def _write_text(text: str, output: str | None) -> None:
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run inference on: cpu, cuda, or mps (default: cpu).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format for batch results (default: json).",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Path to write output to (default: stdout)."
    )


def _cmd_detect_mentions(args: argparse.Namespace) -> None:
    detector = GroupAppealDetector(device=args.device)
    if args.file:
        texts = _read_lines(args.file)
        result = detector.detect_mentions_batch(
            texts, batch_size=args.batch_size, as_df=(args.format == "csv")
        )
    elif args.text:
        result = detector.detect_mentions(args.text)
    else:
        raise SystemExit("Provide TEXT or --file.")
    _write_result(result, args.output, args.format)


def _cmd_classify_stance(args: argparse.Namespace) -> None:
    detector = GroupAppealDetector(device=args.device)
    if args.file:
        df = pd.read_csv(args.file)
        pairs = list(zip(df["text"], df["target_group"]))
        result = detector.classify_stance_batch(
            pairs, batch_size=args.batch_size, as_df=(args.format == "csv")
        )
    elif args.text and args.target_group:
        result = detector.classify_stance(args.text, args.target_group)
    else:
        raise SystemExit("Provide TEXT and TARGET_GROUP, or --file.")
    _write_result(result, args.output, args.format)


def _cmd_detect(args: argparse.Namespace) -> None:
    detector = GroupAppealDetector(device=args.device)
    if args.file:
        texts = _read_lines(args.file)
        result = detector.detect_batch(
            texts, batch_size=args.batch_size, as_df=(args.format == "csv")
        )
    elif args.text:
        result = detector.detect(args.text)
    else:
        raise SystemExit("Provide TEXT or --file.")
    _write_result(result, args.output, args.format)


def _cmd_cluster(args: argparse.Namespace) -> None:
    mentions = args.mentions if args.mentions else (_read_lines(args.file) if args.file else [])
    if not mentions:
        raise SystemExit("Provide one or more MENTIONs or --file.")
    clusterer = GroupMentionClusterer(mentions, device=args.device)
    result = clusterer.cluster(args.n_clusters, as_df=(args.format == "csv"))
    _write_result(result, args.output, args.format)


def _cmd_find_optimal_k(args: argparse.Namespace) -> None:
    mentions = args.mentions if args.mentions else (_read_lines(args.file) if args.file else [])
    if not mentions:
        raise SystemExit("Provide one or more MENTIONs or --file.")
    clusterer = GroupMentionClusterer(mentions, device=args.device)
    dictionary_df = pd.read_csv(args.dictionary) if args.dictionary else None
    best_k, scores = clusterer.find_optimal_k(
        k_range=(args.k_min, args.k_max),
        metric=args.metric,
        dictionary_df=dictionary_df,
        visualize=args.visualize,
    )
    _write_text(json.dumps({"best_k": best_k, "scores": scores}, indent=2), args.output)


def _add_mentions_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mentions", nargs="*", help="Social group mention strings to cluster.")
    parser.add_argument(
        "--file", default=None, help="Path to a file with one mention per line."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run inference on: cpu, cuda, or mps (default: cpu).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="group-appeal-detector",
        description="Detect social group mentions and classify stances toward them.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_detect = subparsers.add_parser(
        "detect", help="Detect group mentions and classify stance toward each."
    )
    p_detect.add_argument("text", nargs="?", help="Input text to analyze.")
    p_detect.add_argument("--file", default=None, help="Path to a file with one text per line.")
    p_detect.add_argument(
        "--batch-size", type=int, default=32, help="Batch size when using --file (default: 32)."
    )
    _add_common_args(p_detect)
    p_detect.set_defaults(func=_cmd_detect)

    p_mentions = subparsers.add_parser(
        "detect-mentions", help="Detect social group mentions in text."
    )
    p_mentions.add_argument("text", nargs="?", help="Input text to analyze.")
    p_mentions.add_argument(
        "--file", default=None, help="Path to a file with one text per line."
    )
    p_mentions.add_argument(
        "--batch-size", type=int, default=32, help="Batch size when using --file (default: 32)."
    )
    _add_common_args(p_mentions)
    p_mentions.set_defaults(func=_cmd_detect_mentions)

    p_stance = subparsers.add_parser(
        "classify-stance", help="Classify stance toward a group mentioned in text."
    )
    p_stance.add_argument("text", nargs="?", help="Input text to analyze.")
    p_stance.add_argument("target_group", nargs="?", help="Social group to classify stance toward.")
    p_stance.add_argument(
        "--file",
        default=None,
        help="Path to a CSV file with 'text' and 'target_group' columns.",
    )
    p_stance.add_argument(
        "--batch-size", type=int, default=32, help="Batch size when using --file (default: 32)."
    )
    _add_common_args(p_stance)
    p_stance.set_defaults(func=_cmd_classify_stance)

    p_cluster = subparsers.add_parser("cluster", help="Cluster social group mentions.")
    _add_mentions_args(p_cluster)
    p_cluster.add_argument(
        "--n-clusters", type=int, required=True, help="Number of clusters to produce."
    )
    p_cluster.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json).",
    )
    p_cluster.add_argument(
        "-o", "--output", default=None, help="Path to write output to (default: stdout)."
    )
    p_cluster.set_defaults(func=_cmd_cluster)

    p_optimal_k = subparsers.add_parser(
        "find-optimal-k", help="Find the optimal number of clusters for a set of mentions."
    )
    _add_mentions_args(p_optimal_k)
    p_optimal_k.add_argument("--k-min", type=int, default=2, help="Minimum k to evaluate (default: 2).")
    p_optimal_k.add_argument("--k-max", type=int, default=30, help="Maximum k to evaluate (default: 30).")
    p_optimal_k.add_argument(
        "--metric",
        choices=["silhouette", "nmi"],
        default="silhouette",
        help="Validation metric to use (default: silhouette).",
    )
    p_optimal_k.add_argument(
        "--dictionary",
        default=None,
        help="Path to a CSV dictionary file, required when --metric=nmi.",
    )
    p_optimal_k.add_argument(
        "--visualize", action="store_true", help="Show a plot of the metric across k values."
    )
    p_optimal_k.add_argument(
        "-o", "--output", default=None, help="Path to write output to (default: stdout)."
    )
    p_optimal_k.set_defaults(func=_cmd_find_optimal_k)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except GroupAppealDetectorError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
