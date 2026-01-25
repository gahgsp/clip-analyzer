from fastapi import APIRouter

from app.api.router import v1 as v1_router

router = APIRouter()

router.include_router(router=v1_router)
