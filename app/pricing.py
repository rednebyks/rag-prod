# USD per 1M tokens — single source of truth for cost calculations
PRICING: dict[str, dict[str, float]] = {
    "meta-llama/llama-3.1-8b-instruct:free": {"input": 0.0, "output": 0.0},
    "google/gemini-2.0-flash-exp:free": {"input": 0.0, "output": 0.0},
    "meta-llama/llama-3.2-3b-instruct:free": {"input": 0.0, "output": 0.0},
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.06, "output": 0.06},
    "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
    "mistralai/mistral-7b-instruct": {"input": 0.055, "output": 0.055},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "mistralai/mistral-large": {"input": 2.00, "output": 6.00},
}

_DEFAULT = {"input": 1.0, "output": 3.0}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICING.get(model, _DEFAULT)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
