
def estimate_token_count(text: str) -> int:
    """
    Rough token estimate: ~4 chars per token for English text.
    Used to bail out early when input exceeds model context window.
    """
    return max(1, len(text) // 4)
