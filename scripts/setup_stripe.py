#!/usr/bin/env python3
"""Create a Stripe Payment Link and wire it into .env for the hackathon."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from simulator.stripe_pay import create_payment_link

ENV_PATH = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"


def _upsert_env(key: str, value: str) -> None:
    line = f"{key}={value}\n"
    if ENV_PATH.exists():
        text = ENV_PATH.read_text(encoding="utf-8")
        pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(line.rstrip(), text)
        else:
            if text and not text.endswith("\n"):
                text += "\n"
            text += "\n# Stripe (hackathon payments)\n" + line
        ENV_PATH.write_text(text, encoding="utf-8")
    else:
        ENV_PATH.write_text(line, encoding="utf-8")


def main() -> int:
    load_dotenv(ENV_PATH)
    parser = argparse.ArgumentParser(description="Create Stripe Payment Link for hackathon billing")
    parser.add_argument(
        "--team",
        default=os.environ.get("STRIPE_TEAM_NAME") or "Shopper Panel",
        help="Product name shown on the Payment Link (e.g. your team name)",
    )
    parser.add_argument(
        "--price",
        type=int,
        default=0,
        help="Fixed price in cents (default: customer chooses price)",
    )
    parser.add_argument(
        "--secret-key",
        default=os.environ.get("STRIPE_SECRET_KEY") or "",
        help="Stripe secret key (sk_test_... or sk_live_...). Not stored in .env.",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        default=True,
        help="Write STRIPE_PAYMENT_LINK to .env (default: on)",
    )
    parser.add_argument(
        "--no-write-env",
        action="store_false",
        dest="write_env",
        help="Print the link only; do not modify .env",
    )
    args = parser.parse_args()

    secret = (args.secret_key or "").strip()
    if not secret:
        print(
            "Missing Stripe secret key.\n\n"
            "1. Sign up at https://dashboard.stripe.com/register\n"
            "2. Developers → API keys → copy Secret key (sk_test_... is fine)\n"
            "3. Re-run:\n"
            "   STRIPE_SECRET_KEY=sk_test_... .venv/bin/python scripts/setup_stripe.py --team \"Your Team\"\n",
            file=sys.stderr,
        )
        return 1

    fixed_price = args.price if args.price > 0 else None
    try:
        url = create_payment_link(
            secret,
            product_name=f"{args.team} payment",
            customer_chooses_price=fixed_price is None,
            unit_amount_cents=fixed_price,
        )
    except Exception as exc:
        print(f"Stripe API error: {exc}", file=sys.stderr)
        return 1

    print(f"\nPayment Link URL:\n{url}\n")
    if args.write_env:
        _upsert_env("STRIPE_PAYMENT_LINK", url)
        print(f"Updated {ENV_PATH} with STRIPE_PAYMENT_LINK")

    print(
        "\nHackathon checklist (do these in Stripe Dashboard):\n"
        "  1. Payment link above — use this same link for every sale today.\n"
        "  2. Developers → API keys → Create restricted key (Read: Balance + Charges).\n"
        "  3. Submit to organizers: team name, Payment Link URL, restricted key (rk_...).\n"
        "\nDo NOT put STRIPE_SECRET_KEY in .env or share it with organizers.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
