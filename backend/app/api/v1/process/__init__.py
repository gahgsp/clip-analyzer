from fastapi import APIRouter

from app.api.v1.process.process import router as process_router

router = APIRouter(prefix='/process')

router.include_router(router=process_router)
