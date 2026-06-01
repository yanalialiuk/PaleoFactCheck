import argparse

from data_processing.data_builder import build_dataset
from fact_check import fact_check

DEFAULT_QUERY = (
    "Характерной особенностью анкилозавров являются костные образования на туловище."
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

    print("Проверка факта...")
    result = fact_check(args.query)
    print("Результат:")
    print(result)


if __name__ == "__main__":
    main()