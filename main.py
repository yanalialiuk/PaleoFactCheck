import argparse

from data_processing.data_builder import build_dataset
from fact_check import DEFAULT_TOP_K, run_fact_check

DEFAULT_QUERY = (
    "A defining feature of ankylosaurs is bony armor along the body."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local paleontology fact check.")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="Claim to verify")
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Rebuild the Chroma index from source documents before checking",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of Chroma chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if args.build_dataset:
        build_dataset()

    print("Checking claim...")
    result = run_fact_check(args.query, top_k=args.top_k)
    print(f"Verdict: {result.verdict}")
    if result.sources:
        print(f"Sources: {', '.join(result.sources)}")
    print("Result:")
    print(result.as_text())


if __name__ == "__main__":
    main()