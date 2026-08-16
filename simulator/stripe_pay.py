"""Stripe payment link helpers for hackathon billing."""

from __future__ import annotations

import os


def stripe_configured() -> bool:
    return bool((os.environ.get("STRIPE_PAYMENT_LINK") or "").strip())


def payment_link() -> str:
    return (os.environ.get("STRIPE_PAYMENT_LINK") or "").strip()


def create_payment_link(
    secret_key: str,
    *,
    product_name: str = "Shopper panel payment",
    customer_chooses_price: bool = True,
    unit_amount_cents: int | None = None,
    currency: str = "usd",
) -> str:
    """Create a Stripe Payment Link (Dashboard-equivalent via API)."""
    import stripe

    stripe.api_key = secret_key
    product = stripe.Product.create(name=product_name)
    price_params: dict = {"product": product.id, "currency": currency}
    if customer_chooses_price:
        price_params["custom_unit_amount"] = {"enabled": True}
    else:
        price_params["unit_amount"] = unit_amount_cents or 2900
    price = stripe.Price.create(**price_params)
    link = stripe.PaymentLink.create(line_items=[{"price": price.id, "quantity": 1}])
    return link.url
