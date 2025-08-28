
def sliding_window_chunks(text, chunk_size=512, overlap=64):
    """Split text into overlapping windows for better cross-chunk retrieval."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(' '.join(words[i:i+chunk_size]))
        i += (chunk_size - overlap)
    return chunks
