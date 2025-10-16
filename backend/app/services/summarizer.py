
_EXTRACTIVE_MAX_SENTENCES = 5

def extractive_fallback(text):
    """
    Extractive summary: return the first N sentences when the LLM is unavailable.
    Not as good, but always available offline.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:_EXTRACTIVE_MAX_SENTENCES])
