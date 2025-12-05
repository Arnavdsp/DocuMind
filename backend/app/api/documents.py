
# Pagination: default page_size=20, max=100
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100

def _paginate(items, page, page_size):
    start = (page - 1) * page_size
    return items[start:start + page_size]
