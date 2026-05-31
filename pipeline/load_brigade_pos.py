"""
Load actual Brigade Road POS data (ST1008 format) into the Store Intelligence API.

The Brigade CSV has one row per line-item, not per transaction. This script
aggregates rows by invoice_number to reconstruct per-transaction basket values,
then POSTs them to /pos/ingest.

CSV columns used:
  invoice_number  — unique transaction identifier (maps to transaction_id)
  order_date      — "DD-MM-YYYY" format
  order_time      — "HH:MM:SS" format
  total_amount    — per-line-item amount (summed per invoice_number)
  store_id        — "ST1008" in source CSV (remapped via --store-id flag)

Usage:
    python -m pipeline.load_brigade_pos \\
        --csv "Brigade_Bangalore_10_April_26.csv" \\
        --api-url http://localhost:8000 \\
        --store-id STORE_BLR_002
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


def _parse_timestamp(order_date: str, order_time: str) -> str:
    """Parse 'DD-MM-YYYY' + 'HH:MM:SS' → ISO-8601 UTC string."""
    dt = datetime.strptime(f"{order_date} {order_time}", "%d-%m-%Y %H:%M:%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")  # Brigade Road is IST (UTC+5:30)


def load_brigade_csv(csv_path: str, api_url: str, target_store_id: str) -> None:
    """
    Aggregate Brigade CSV line-items by invoice_number and POST to /pos/ingest.

    Handles:
    - Multiple line-items per invoice (grouped by invoice_number)
    - Zero-value carry-bag items (excluded from basket total)
    - Return transactions (invoice_type == 'return', basket_value set negative)
    """
    # invoice_number → {timestamp, basket_value, is_return}
    invoices: dict[str, dict] = defaultdict(lambda: {"timestamp": None, "basket_value": 0.0, "is_return": False})

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inv = row.get("invoice_number", "").strip()
            if not inv:
                continue

            inv_type = row.get("invoice_type", "sales").strip()
            order_date = row.get("order_date", "").strip()
            order_time = row.get("order_time", "").strip()

            # Parse timestamp on first encounter of this invoice
            if invoices[inv]["timestamp"] is None and order_date and order_time:
                try:
                    invoices[inv]["timestamp"] = _parse_timestamp(order_date, order_time)
                except ValueError:
                    logger.warning("Bad date/time for invoice %s: %s %s", inv, order_date, order_time)

            # Sum basket value — skip carry bags (total_amount == 0 or tiny)
            try:
                amount = float(row.get("total_amount", 0) or 0)
            except ValueError:
                amount = 0.0

            if amount > 0.5:  # exclude free/gift items with negligible value
                invoices[inv]["basket_value"] += amount

            if inv_type == "return":
                invoices[inv]["is_return"] = True

    # Build transaction list
    transactions = []
    skipped = 0
    for inv_number, data in invoices.items():
        if data["timestamp"] is None or data["basket_value"] <= 0:
            skipped += 1
            continue
        basket = data["basket_value"]
        if data["is_return"]:
            basket = -basket  # negative basket signals a return
        transactions.append({
            "transaction_id": f"BLR_{inv_number}",
            "store_id": target_store_id,
            "timestamp": data["timestamp"],
            "basket_value_inr": round(basket, 2),
        })

    if not transactions:
        logger.error("No valid transactions found in %s (skipped=%d)", csv_path, skipped)
        sys.exit(1)

    logger.info("Parsed %d invoices (skipped %d), posting to %s", len(transactions), skipped, api_url)

    # POST in batches of 500
    total_loaded = 0
    for i in range(0, len(transactions), 500):
        batch = transactions[i: i + 500]
        try:
            resp = requests.post(
                f"{api_url}/pos/ingest",
                json={"transactions": batch},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            total_loaded += result.get("loaded", 0)
            logger.info(
                "Batch %d: loaded=%d duplicates=%d",
                i // 500 + 1,
                result.get("loaded", 0),
                result.get("duplicates", 0),
            )
        except Exception as exc:
            logger.error("Failed to load batch %d: %s", i // 500 + 1, exc)
            sys.exit(1)

    print(
        f"Brigade POS load complete.\n"
        f"  Transactions loaded : {total_loaded}\n"
        f"  Store ID            : {target_store_id}\n"
        f"  API                 : {api_url}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Load Brigade Road POS CSV into the Store Intelligence API")
    parser.add_argument("--csv", required=True, help="Path to Brigade_Bangalore_*.csv")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--store-id", default="STORE_BLR_002",
                        help="Target store_id in the API (default: STORE_BLR_002)")
    args = parser.parse_args()
    load_brigade_csv(args.csv, args.api_url, args.store_id)


if __name__ == "__main__":
    main()
