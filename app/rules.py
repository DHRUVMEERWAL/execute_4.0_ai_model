# app/rules.py

def apply_business_rules(transaction):
    triggered = False
    reasons = []

    # Rule 1: High transaction amount
    if transaction['transaction_amount'] > 100000:
        triggered = True
        reasons.append("rule: high_amount")

    # Rule 2: Night-time risky transaction on channel 0
    if transaction['transaction_channel'] == 0:
        from datetime import datetime
        txn_hour = datetime.fromisoformat(transaction['transaction_date']).hour
        if txn_hour in [0, 1, 2, 3, 4]:
            triggered = True
            reasons.append("rule: night_time_channel_0")

    return {
        "triggered": triggered,
        "reasons": reasons
    }
