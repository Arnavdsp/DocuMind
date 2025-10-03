
def deduplicate_chunks(chunks, threshold=0.95):
    """
    Remove near-duplicate chunks by Jaccard overlap on token sets.
    threshold=0.95 means chunks sharing 95%+ tokens are considered duplicates.
    """
    seen, unique = [], []
    for c in chunks:
        tokens = frozenset(c.get('text','').lower().split())
        if not any(len(tokens & s) / max(len(tokens | s), 1) > threshold for s in seen):
            seen.append(tokens)
            unique.append(c)
    return unique
