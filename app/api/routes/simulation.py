from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_simulations():
    return {"msg": "Simulation Data"}
