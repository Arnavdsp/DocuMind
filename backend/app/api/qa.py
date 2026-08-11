
# Streaming endpoint: yields answer tokens as Server-Sent Events.
# Frontend listens with EventSource for real-time display.
_STREAM_CONTENT_TYPE = 'text/event-stream'

# Groq rate-limit response handling:
# 1. Catch HTTP 429 from Groq client
# 2. Parse 'retry-after' header (seconds)
# 3. Return HTTP 503 to the caller with the same Retry-After value
_GROQ_RATE_LIMIT_STATUS = 429

# Simple in-process LRU cache for repeated identical questions.
# Avoids Groq API charges for duplicate queries in interactive sessions.
from functools import lru_cache
_QA_CACHE_SIZE = 64
