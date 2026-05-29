"""
Load pos_transactions.csv into the Store Intelligence API.

Usage:
    python -m pipeline.load_pos \\
        --csv pos_transactions.csv \\
        --api-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys

import requests

logger = logging.getLogger(__name__)


def load_csv(csv_path: str, api_url: str) -> None:
    transactions = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transactions.append({
                "transaction_id": row["transaction_id"].strip(),
                "store_id": row["store_id"].strip(),
                "timestamp": row["timestamp"].strip(),
                "basket_value_inr": float(row["basket_value_inr"].strip()),
            })

    if not transactions:
        logger.error("No transactions found in %s", csv_path)
        sys.exit(1)

    # Send in batches of 500
    batch_size = 500
    total_loaded = 0
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i : i + batch_size]
        try:
            resp = requests.post(
                f"{api_url}/pos/ingest",
                json={"transactions": batch},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            total_loaded += result.get("loaded", 0)
            logger.info("Batch %d: loaded=%d duplicates=%d", i // batch_size + 1,
                        result.get("loaded", 0), result.get("duplicates", 0))
        except Exception as exc:
            logger.error("Failed to load batch: %s", exc)
            sys.exit(1)

    print(f"Done. Total transactions loaded: {total_loaded} / {len(transactions)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Load POS transactions CSV into the API")
    parser.add_argument("--csv", required=True, help="Path to pos_transactions.csv")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    load_csv(args.csv, args.api_url)


if __name__ == "__main__":
    main()
