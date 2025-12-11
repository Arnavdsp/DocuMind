
# Pagination: default page_size=20, max=100
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100

def _paginate(items, page, page_size):
    start = (page - 1) * page_size
    return items[start:start + page_size]

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# Uploads larger than this return HTTP 422 with a human-readable error
# rather than letting the server OOM trying to embed a 200 MB scan.
