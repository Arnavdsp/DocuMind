
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
