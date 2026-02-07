import logging
import sys
from fastapi import FastAPI
from app.router import router as master_router

# Configuring the root logger so we can add our own log messages without relying
# in the Uvicorn log system.
# Here we configure the level to INFO, we add a formatter on how we want to display the logs
# plus defining where we want to display the custom log messages (handler).
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = FastAPI(title="Clip Analyzer API")

app.include_router(router=master_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="app.main:app", host="0.0.0.0", port=8000, reload=True)
