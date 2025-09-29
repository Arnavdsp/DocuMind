
_EMBED_BATCH_SIZE = 32

def batch_embed(texts, embed_fn):
    """Call embed_fn in batches to control peak memory usage."""
    results = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        results.extend(embed_fn(texts[i:i+_EMBED_BATCH_SIZE]))
    return results
