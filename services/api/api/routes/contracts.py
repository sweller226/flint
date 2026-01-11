from fastapi import APIRouter
from api.api.dependencies import get_available_contracts

router = APIRouter(tags=["contracts"])


@router.get("/contracts")
async def list_contracts():
    """Get list of available ES futures contracts."""
    return {
        "contracts": get_available_contracts(),
        "default": "H"
    }
