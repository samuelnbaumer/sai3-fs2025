import base64
import tempfile
import os

from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

root = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()
# Allow all origins for simplicity; adjust as needed
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://0.0.0.0"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/url/{request_text}", response_class=PlainTextResponse)
async def get_url(request_text):
    return request_text


@app.get("/")
async def main():
    # print(root)
    with open(os.path.join(root, 'frontend.html')) as fh:
        data = fh.read()
    return Response(content=data, media_type="text/html")
