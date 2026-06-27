"""Price tables for cost tracking. Prices are USD per 1M tokens.

Verify against current OpenAI pricing periodically — these change. Local
models cost $0, so they map to zero here.
"""

PRICES : dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10),
    "text-embedding-3-small": (0.22, 0.00),
    "text-embedding-3-large": (0.13, 0.00),

}

def cost_usd(model:str , input_tokens:int, output_tokens:int) -> float:
    """Return the cost in USD for a given model and token usage."""
    if model not in PRICES:
        raise ValueError(f"Unknown model {model}")
    input_price, output_price = PRICES[model]
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price