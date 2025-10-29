
_EXTRACTIVE_MAX_SENTENCES = 5

def extractive_fallback(text):
    """
    Extractive summary: return the first N sentences when the LLM is unavailable.
    Not as good, but always available offline.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:_EXTRACTIVE_MAX_SENTENCES])

_GROQ_MAX_INPUT_TOKENS = 8192
_SUMMARY_INPUT_CAP = 6000  # leave room for system prompt + output

def truncate_for_groq(text, cap=_SUMMARY_INPUT_CAP):
    words = text.split()
    return ' '.join(words[:cap]) if len(words) > cap else text

SUMMARY_MODES = ('paragraph', 'bullets', 'tldr')

BULLET_SYSTEM_PROMPT = (
    "Summarise the document as a concise bulleted list. "
    "Each bullet should be one sentence. "
    "Return plain text with one bullet per line starting with '- '."
)
