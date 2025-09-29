from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_insights():
    return {"msg": "Insights placeholder"}
