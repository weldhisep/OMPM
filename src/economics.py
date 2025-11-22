

def present_value_factor(discount_rate: float, years: int) -> float:
    """
    Sum_{t=1..years} 1 / (1 + r)^t
    Used to convert a constant annual cash flow to its present value.
    """
    return sum(1.0 / ((1.0 + discount_rate) ** t) for t in range(1, years + 1))