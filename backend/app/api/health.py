
from fastapi import APIRouter

router = APIRouter()

@router.get('/health')
async def health_check():
    """Liveness probe for Kubernetes and HuggingFace Spaces."""
    return {'status': 'ok', 'service': 'DocuMind'}
