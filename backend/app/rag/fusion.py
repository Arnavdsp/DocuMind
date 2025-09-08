
def reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60):
    """
    Merge dense and sparse retrieval ranks using RRF.
    Returns a dict of doc_id -> combined score.
    """
    scores = {}
    for rank_list in (dense_ranks, sparse_ranks):
        for rank, doc_id in enumerate(rank_list, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: -x[1]))
