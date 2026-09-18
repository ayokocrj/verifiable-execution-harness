# -*- coding: utf-8 -*-
"""Draft the outreach message. Templates by cause; optional LLM polish if a key is set.

Set ANTHROPIC_API_KEY (and `pip install anthropic`) to get LLM-personalised drafts.
Without a key the templates are used, so the demo always runs.
"""
import os

TEMPLATES = {
    "card_expired": (
        "Hi {first}, the card on file for {company} expired, so this month's {amount:.0f} $ payment didn't go through. "
        "Update it here in 30 seconds: {link}. Your account stays active in the meantime."
    ),
    "insufficient_funds": (
        "Hi {first}, your {amount:.0f} $ payment for {company} was declined by the bank. "
        "We'll retry automatically in 3 days; if you'd rather use another card, update it here: {link}."
    ),
    "card_changed": (
        "Hi {first}, it looks like the card for {company} was replaced. "
        "Add the new one here so scheduling and invoicing keep running: {link}."
    ),
    "unknown": (
        "Hi {first}, this month's {amount:.0f} $ payment for {company} didn't complete. "
        "Could you check the card on file here: {link}? Reply to this email if anything looks off."
    ),
    "churn_signal": (
        "Hi {first}, I noticed {company} has been quieter on the platform lately ({signals}). "
        "Is something blocking your team? Happy to jump on a 15-minute call this week to sort it out."
    ),
}


def _llm_polish(text):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return text
    try:
        import anthropic  # optional dependency
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": f"Rewrite this customer email to be warm, concise and specific. Keep all facts and the link. Return only the email body.\n\n{text}"}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return text


def draft(case):
    key = case["cause"] if case["kind"] == "failed_payment" else "churn_signal"
    text = TEMPLATES[key].format(
        first="there",
        company=case["company"],
        amount=case.get("amount", case["mrr"]),
        link=f"https://billing.example.com/update/{case['customer_id']}",
        signals=", ".join(case.get("signals", [])),
    )
    return _llm_polish(text)
