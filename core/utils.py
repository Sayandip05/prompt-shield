def calculate_platform_cut(amount: float, cut_percent: float = 10.0) -> float:
    return round((amount * cut_percent) / 100.0, 2)


def format_currency(amount: float, currency: str = "USD") -> str:
    return f"{currency} {amount:,.2f}"
