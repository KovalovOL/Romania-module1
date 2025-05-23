from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.nover_router import router as novel_router


app = FastAPI()
app.include_router(novel_router, prefix="/novel")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return("Root")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)