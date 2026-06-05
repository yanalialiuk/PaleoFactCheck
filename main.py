import argparse

from data_processing.data_builder import build_dataset
from fact_check import run_fact_check

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
    args = parser.parse_args()

    if args.build_dataset:
        build_dataset()

    print("Checking claim...")
    result = run_fact_check(args.query)
    print(f"Verdict: {result.verdict}")
    if result.sources:
        print(f"Sources: {', '.join(result.sources)}")
    print("Result:")
    print(result.as_text())


if __name__ == "__main__":
    main()