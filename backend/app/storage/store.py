
async def delete_document(doc_id: str, vector_store, metadata_db) -> bool:
    """
    Delete a document and all its associated chunk vectors.
    Returns True if deleted, False if doc_id was not found.
    """
    exists = await metadata_db.exists(doc_id)
    if not exists:
        return False
    chunk_ids = await metadata_db.get_chunk_ids(doc_id)
    vector_store.delete(chunk_ids)
    await metadata_db.delete(doc_id)
    return True

# HuggingFace Spaces: uploaded documents are private to this Space instance.
# Never expose raw file URLs in API responses — serve through /documents/{id}/download.
_STORAGE_VISIBILITY = 'private'
