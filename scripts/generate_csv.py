#!/usr/bin/env python3
"""
Generate a supplier CSV with random data for testing vector search.
Uses Faker to produce ~30 columns of varied supplier information.
"""

import argparse
import csv
from pathlib import Path

from faker import Faker

# Column definitions: (name, faker_callable or static values)
FAKE = Faker()

COLUMNS = [
    # Identity
    ("supplier_id", lambda: f"SUP-{FAKE.unique.random_int(min=10000, max=99999)}"),
    ("supplier_name", lambda: FAKE.company()),
    ("trading_name", lambda: FAKE.company()),
    ("company_registration_number", lambda: FAKE.bothify(text="??######")),
    # Address
    ("address_line_1", lambda: FAKE.street_address()),
    ("address_line_2", lambda: FAKE.secondary_address() if FAKE.boolean(chance_of_getting_true=30) else ""),
    ("city", lambda: FAKE.city()),
    ("state_province", lambda: FAKE.state_abbr()),
    ("postal_code", lambda: FAKE.postcode()),
    ("country", lambda: FAKE.country_code()),
    # Contact
    ("contact_first_name", lambda: FAKE.first_name()),
    ("contact_last_name", lambda: FAKE.last_name()),
    ("contact_email", lambda: FAKE.company_email()),
    ("contact_phone", lambda: FAKE.phone_number()),
    ("website", lambda: FAKE.domain_name()),
    # Products
    ("product_code_1", lambda: FAKE.bothify(text="PRD-####")),
    ("product_name_1", lambda: FAKE.catch_phrase()),
    ("product_category_1", lambda: FAKE.word().capitalize()),
    ("product_code_2", lambda: FAKE.bothify(text="PRD-####")),
    ("product_name_2", lambda: FAKE.bs()),
    ("product_category_2", lambda: FAKE.word().capitalize()),
    ("product_code_3", lambda: FAKE.bothify(text="PRD-####")),
    ("product_name_3", lambda: FAKE.catch_phrase()),
    ("product_category_3", lambda: FAKE.word().capitalize()),
    # Business
    ("commodity", lambda: FAKE.random_element(["Electronics", "Industrial", "Automotive", "Chemicals", "Food & Beverage", "Construction", "Healthcare", "Textiles", "Metals", "Plastics"])),
    ("industry_sector", lambda: FAKE.random_element(["Manufacturing", "Wholesale", "Retail", "Distribution", "Services", "Technology", "Engineering"])),
    ("annual_revenue", lambda: str(FAKE.random_int(min=100000, max=50000000))),
    ("employee_count", lambda: str(FAKE.random_int(min=5, max=5000))),
    ("year_established", lambda: str(FAKE.random_int(min=1980, max=2024))),
    ("certification", lambda: FAKE.random_element(["ISO 9001", "ISO 14001", "ISO 27001", "FDA", "CE", "None", "ISO 9001, ISO 14001"])),
    ("payment_terms", lambda: FAKE.random_element(["Net 30", "Net 60", "Net 15", "COD", "50% advance"])),
    ("lead_time_days", lambda: str(FAKE.random_int(min=1, max=90))),
    ("minimum_order_value", lambda: str(FAKE.random_int(min=100, max=10000))),
    ("currency", lambda: FAKE.random_element(["USD", "EUR", "GBP"])),
    # Misc
    ("description", lambda: FAKE.paragraph(nb_sentences=2)),
    ("tags", lambda: ", ".join(FAKE.words(nb=3))),
]


def generate_row() -> dict[str, str]:
    """Generate a single row of supplier data."""
    row = {}
    for col_name, generator in COLUMNS:
        try:
            val = generator()
            row[col_name] = "" if val is None else str(val).strip()
        except Exception:
            FAKE.unique.clear()
            val = generator()
            row[col_name] = "" if val is None else str(val).strip()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate supplier CSV with random data")
    parser.add_argument("-n", "--rows", type=int, default=10_000, help="Number of rows to generate (default: 10000)")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/suppliers.csv"), help="Output CSV path")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    headers = [c[0] for c in COLUMNS]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for _ in range(args.rows):
            writer.writerow(generate_row())

    print(f"Generated {args.rows} rows to {args.output}")


if __name__ == "__main__":
    main()
