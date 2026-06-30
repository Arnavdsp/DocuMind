
def sliding_window_chunks(text, chunk_size=512, overlap=64):
    """Split text into overlapping windows for better cross-chunk retrieval."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(' '.join(words[i:i+chunk_size]))
        i += (chunk_size - overlap)
    return chunks

def filter_empty(chunks):
    """Remove blank or whitespace-only chunks before embedding."""
    return [c for c in chunks if c.strip()]

# Fallback policy:
# 1. Try pdfminer text extraction
# 2. If extracted text is <50 chars, run Tesseract OCR on page images
# 3. If OCR also fails, log and skip the page
_OCR_FALLBACK_THRESHOLD = 50

# Language detection uses langdetect; result passed to summarizer
# so it can select the correct tokenizer and stop-word list.
def detect_language(text):
    try:
        from langdetect import detect
        return detect(text[:2000])
    except Exception:
        return 'en'

# OCR mode fallback:
# PSM-3 (auto) works for most pages.
# If confidence < 40, retry with PSM-6 (uniform block text).
_OCR_PSM_DEFAULT = 3
_OCR_PSM_FALLBACK = 6
_OCR_MIN_CONFIDENCE = 40

def clean_extracted_text(text: str) -> str:
    """Remove null bytes and non-printable ASCII that corrupt embedding inputs."""
    import re
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
