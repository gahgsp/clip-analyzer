from fastapi import APIRouter
from app.api.v1.process import router as process_router

v1 = APIRouter(prefix="/v1")

v1.include_router(router=process_router)
