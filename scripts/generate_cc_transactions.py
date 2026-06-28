"""
scripts/generate_cc_transactions.py — Generate a synthetic credit card
transaction dataset and save it to data/credit_card_transactions.csv.

Design goals:
  - 2.5 million rows, fully vectorised (fast — ~60s)
  - Realistic distributions: log-normal spend, fraud patterns, category splits
  - Rich enough for interesting queries:
      - Fraud rate by card type / merchant category / country
      - Spend trends by hour, day, month
      - Top merchants, highest-value transactions
      - Declined vs approved vs reversed by category
      - Cardholder age vs average spend

Usage:
    python -m scripts.generate_cc_transactions
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ── Config ─────────────────────────────────────────────────────────────────────

N_ROWS          = 2_500_000
CHUNK_SIZE      = 500_000       # write in chunks to keep memory low
N_CARDHOLDERS   = 80_000        # unique cardholders
N_MERCHANTS     = 8_000         # unique merchants
RANDOM_SEED     = 42
OUTPUT_PATH     = Path("data/credit_card_transactions.csv")

fake = Faker("en_US")
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Categorical pools ──────────────────────────────────────────────────────────

CARD_TYPES      = ["Visa", "Mastercard", "Amex", "Discover"]
CARD_WEIGHTS    = [0.45,   0.35,          0.12,   0.08]

CATEGORIES      = [
    "Groceries", "Restaurants", "Online Retail", "Travel",
    "Entertainment", "Gas & Fuel", "Healthcare", "Electronics",
    "Clothing", "Other",
]
CAT_WEIGHTS     = [0.20, 0.17, 0.16, 0.11, 0.09, 0.08, 0.07, 0.05, 0.04, 0.03]

# Typical spend range per category: (mu, sigma) for log-normal
CAT_SPEND = {
    "Groceries":      (3.8, 0.6),   # ~$45 median
    "Restaurants":    (3.5, 0.5),   # ~$33
    "Online Retail":  (4.0, 0.7),   # ~$55
    "Travel":         (5.2, 0.9),   # ~$181
    "Entertainment":  (3.6, 0.6),   # ~$37
    "Gas & Fuel":     (3.7, 0.4),   # ~$40
    "Healthcare":     (4.3, 0.8),   # ~$74
    "Electronics":    (5.0, 0.8),   # ~$148
    "Clothing":       (4.0, 0.6),   # ~$55
    "Other":          (3.9, 0.7),   # ~$49
}

# Fraud rate per category (base = 1.5%)
CAT_FRAUD_RATE = {
    "Groceries":      0.007,
    "Restaurants":    0.010,
    "Online Retail":  0.028,
    "Travel":         0.035,
    "Entertainment":  0.018,
    "Gas & Fuel":     0.022,
    "Healthcare":     0.008,
    "Electronics":    0.040,
    "Clothing":       0.020,
    "Other":          0.015,
}

TXN_TYPES       = ["Purchase", "Refund", "Cash Advance"]
TXN_WEIGHTS     = [0.91,       0.07,     0.02]

CURRENCIES      = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "INR", "MXN"]
CURR_WEIGHTS    = [0.72,  0.08,  0.06,  0.05,  0.04,  0.02,  0.02,  0.01]

US_CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "Indianapolis", "Seattle", "Denver", "Nashville", "Boston",
    "Las Vegas", "Memphis", "Portland", "Atlanta", "Miami",
]
INTL_CITIES = [
    "London", "Paris", "Tokyo", "Sydney", "Toronto", "Berlin",
    "Amsterdam", "Singapore", "Dubai", "Mumbai", "São Paulo", "Mexico City",
]

MERCHANT_CITIES   = US_CITIES * 4 + INTL_CITIES
MERCHANT_COUNTRIES = ["United States"] * len(US_CITIES) * 4 + [
    "United Kingdom", "France", "Japan", "Australia", "Canada", "Germany",
    "Netherlands", "Singapore", "UAE", "India", "Brazil", "Mexico",
]

CH_COUNTRIES = [
    "United States", "United Kingdom", "Canada", "Australia", "Germany",
    "France", "Japan", "India", "Brazil", "Mexico",
    "Singapore", "Netherlands", "UAE", "South Korea", "Italy",
]
CH_COUNTRY_WEIGHTS = [
    0.85, 0.03, 0.03, 0.02, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.005,
    0.005, 0.003, 0.003, 0.002, 0.002,
]

# ── Build cardholder pool ──────────────────────────────────────────────────────

def build_cardholders(n: int) -> pd.DataFrame:
    print(f"  Generating {n:,} cardholders...", end=" ", flush=True)
    t = time.time()
    names = [fake.name() for _ in range(n)]
    df = pd.DataFrame({
        "cardholder_id":      range(1, n + 1),
        "cardholder_name":    names,
        "cardholder_age":     np.random.randint(18, 81, size=n),
        "cardholder_city":    np.random.choice(US_CITIES, size=n),
        "cardholder_country": np.random.choice(CH_COUNTRIES, size=n, p=CH_COUNTRY_WEIGHTS),
        "card_type":          np.random.choice(CARD_TYPES, size=n, p=CARD_WEIGHTS),
        "card_last_four":     [f"{random.randint(1000, 9999)}" for _ in range(n)],
        "credit_limit":       np.random.choice(
                                  [500, 1000, 2000, 5000, 10000, 15000, 25000, 50000],
                                  size=n,
                                  p=[0.05, 0.10, 0.15, 0.25, 0.20, 0.12, 0.08, 0.05],
                              ),
    })
    print(f"done ({time.time() - t:.1f}s)")
    return df


# ── Build merchant pool ────────────────────────────────────────────────────────

def build_merchants(n: int) -> pd.DataFrame:
    print(f"  Generating {n:,} merchants...", end=" ", flush=True)
    t = time.time()
    categories = np.random.choice(CATEGORIES, size=n, p=CAT_WEIGHTS)
    city_idx   = np.random.randint(0, len(MERCHANT_CITIES), size=n)
    df = pd.DataFrame({
        "merchant_id":       range(1, n + 1),
        "merchant_name":     [fake.company() for _ in range(n)],
        "merchant_category": categories,
        "merchant_city":     [MERCHANT_CITIES[i] for i in city_idx],
        "merchant_country":  [MERCHANT_COUNTRIES[i] for i in city_idx],
    })
    print(f"done ({time.time() - t:.1f}s)")
    return df


# ── Generate transactions ──────────────────────────────────────────────────────

def generate_chunk(
    start_id: int,
    size: int,
    cardholders: pd.DataFrame,
    merchants: pd.DataFrame,
) -> pd.DataFrame:

    # Sample cardholder and merchant indices
    ch_idx  = np.random.randint(0, len(cardholders), size=size)
    mer_idx = np.random.randint(0, len(merchants),   size=size)

    cats = merchants["merchant_category"].values[mer_idx]

    # Transaction amounts — log-normal per category
    amounts = np.zeros(size)
    for cat in CATEGORIES:
        mask = cats == cat
        mu, sigma = CAT_SPEND[cat]
        n_cat = mask.sum()
        if n_cat > 0:
            amounts[mask] = np.random.lognormal(mu, sigma, size=n_cat)

    # Transaction type
    txn_types = np.random.choice(TXN_TYPES, size=size, p=TXN_WEIGHTS)

    # Cash advances get larger amounts
    ca_mask = txn_types == "Cash Advance"
    amounts[ca_mask] = np.random.lognormal(5.5, 0.6, size=ca_mask.sum())

    # Refunds are negative amounts
    ref_mask = txn_types == "Refund"
    amounts[ref_mask] *= -1

    amounts = amounts.round(2)

    # Fraud flag — vectorised per category
    fraud = np.zeros(size, dtype=bool)
    for cat in CATEGORIES:
        cat_mask = cats == cat
        rate = CAT_FRAUD_RATE[cat]
        fraud[cat_mask] = np.random.random(cat_mask.sum()) < rate

    # Cash advances have higher fraud rate
    fraud[ca_mask] = np.random.random(ca_mask.sum()) < 0.12

    # International transactions (merchant country != US) bump fraud slightly
    mer_countries = merchants["merchant_country"].values[mer_idx]
    intl_mask = mer_countries != "United States"
    extra_fraud = (~fraud) & intl_mask & (np.random.random(size) < 0.03)
    fraud |= extra_fraud

    # Status: fraudulent transactions are declined more often
    status = np.where(
        fraud,
        np.random.choice(
            ["Approved", "Declined", "Reversed"],
            size=size,
            p=[0.35, 0.55, 0.10],
        ),
        np.random.choice(
            ["Approved", "Declined", "Reversed"],
            size=size,
            p=[0.94, 0.05, 0.01],
        ),
    )

    # Timestamps spread across 2022-01-01 to 2023-12-31
    start_ts = pd.Timestamp("2022-01-01").value // 10**9
    end_ts   = pd.Timestamp("2023-12-31 23:59:59").value // 10**9
    unix_ts  = np.random.randint(start_ts, end_ts, size=size)
    # Add hour-of-day bias: more activity 8am-10pm
    hour_bias = np.random.choice(range(24), size=size, p=_hour_weights())
    unix_ts  += hour_bias * 3600
    timestamps = pd.to_datetime(unix_ts, unit="s")

    return pd.DataFrame({
        "transaction_id":      range(start_id, start_id + size),
        "transaction_date":    timestamps,
        "cardholder_id":       cardholders["cardholder_id"].values[ch_idx],
        "cardholder_name":     cardholders["cardholder_name"].values[ch_idx],
        "cardholder_age":      cardholders["cardholder_age"].values[ch_idx],
        "cardholder_city":     cardholders["cardholder_city"].values[ch_idx],
        "cardholder_country":  cardholders["cardholder_country"].values[ch_idx],
        "card_type":           cardholders["card_type"].values[ch_idx],
        "card_last_four":      cardholders["card_last_four"].values[ch_idx],
        "credit_limit":        cardholders["credit_limit"].values[ch_idx],
        "merchant_id":         merchants["merchant_id"].values[mer_idx],
        "merchant_name":       merchants["merchant_name"].values[mer_idx],
        "merchant_category":   cats,
        "merchant_city":       mer_countries,   # reuse variable
        "merchant_country":    mer_countries,
        "transaction_amount":  amounts,
        "currency":            np.random.choice(CURRENCIES, size=size, p=CURR_WEIGHTS),
        "transaction_type":    txn_types,
        "is_fraudulent":       fraud,
        "status":              status,
    }).assign(
        merchant_city=merchants["merchant_city"].values[mer_idx]
    )


def _hour_weights() -> list[float]:
    # Low activity 0-6am, ramp up 7-9am, peak 10am-9pm, taper off
    w = [0.5, 0.3, 0.2, 0.2, 0.2, 0.3,   # 0-5
         0.7, 1.5, 2.5,                    # 6-8
         3.5, 4.0, 4.5, 5.0, 5.0, 5.0,   # 9-14
         5.0, 5.5, 5.5, 5.5, 5.0,         # 15-19
         4.0, 3.0, 2.0, 1.0]              # 20-23
    total = sum(w)
    return [x / total for x in w]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nGenerating {N_ROWS:,} synthetic credit card transactions")
    print(f"Output: {OUTPUT_PATH}\n")
    total_start = time.time()

    cardholders = build_cardholders(N_CARDHOLDERS)
    merchants   = build_merchants(N_MERCHANTS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    first_chunk = True

    for chunk_start in range(0, N_ROWS, CHUNK_SIZE):
        chunk_size = min(CHUNK_SIZE, N_ROWS - chunk_start)
        print(f"  Chunk {chunk_start // CHUNK_SIZE + 1}: rows {chunk_start:,}–{chunk_start + chunk_size:,}...",
              end=" ", flush=True)
        t = time.time()

        chunk = generate_chunk(chunk_start + 1, chunk_size, cardholders, merchants)

        chunk.to_csv(
            OUTPUT_PATH,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        written += chunk_size
        print(f"done ({time.time() - t:.1f}s) — {written:,} rows written")

    elapsed = time.time() - total_start
    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    print(f"\nDone! {written:,} rows in {elapsed:.1f}s")
    print(f"File: {OUTPUT_PATH}  ({size_mb:.1f} MB)")
    print(f"\nColumns: transaction_id, transaction_date, cardholder_id/name/age/city/country,")
    print(f"         card_type, card_last_four, credit_limit, merchant_id/name/category/city/country,")
    print(f"         transaction_amount, currency, transaction_type, is_fraudulent, status")


if __name__ == "__main__":
    main()
