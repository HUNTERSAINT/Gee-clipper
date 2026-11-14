"""Paystack checkout and server-side transaction verification."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class PaymentSetupError(RuntimeError):
    pass


def _api(method, path, payload=None):
    secret = os.getenv("PAYSTACK_SECRET_KEY", "").strip()
    if not secret:
        raise PaymentSetupError("Paystack is not configured. Add PAYSTACK_SECRET_KEY in Railway Variables.")
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        "https://api.paystack.co" + path,
        data=body,
        method=method,
        headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("Paystack request failed. Check payment configuration and try again.") from exc
    if not result.get("status"):
        raise RuntimeError(str(result.get("message", "Paystack rejected the request")))
    return result.get("data") or {}


def initialize(email, user_id):
    raw_amount = os.getenv("PAYSTACK_PREMIUM_PRICE_KOBO", "").strip()
    try:
        amount = int(raw_amount)
    except (TypeError, ValueError):
        amount = 0
    if amount <= 0:
        raise PaymentSetupError("Set PAYSTACK_PREMIUM_PRICE_KOBO to your chosen Premium price in kobo.")
    payload = {
        "email": email,
        "amount": amount,
        "currency": "NGN",
        "metadata": {"telegram_user_id": int(user_id), "product": "Gee Clipper Premium"},
    }
    callback = os.getenv("PAYSTACK_CALLBACK_URL", "").strip()
    if callback:
        payload["callback_url"] = callback
    data = _api("POST", "/transaction/initialize", payload)
    return {"reference": data["reference"], "authorization_url": data["authorization_url"], "amount": amount}


def verify(reference):
    return _api("GET", "/transaction/verify/" + quote(str(reference), safe=""))
