from fastapi import FastAPI
from app.router import router as master_router

app = FastAPI(title="Clip Analyzer API")

app.include_router(router=master_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="app.main:app", host="0.0.0.0", port=8000, reload=True)
