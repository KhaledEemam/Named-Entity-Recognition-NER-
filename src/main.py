from fastapi import FastAPI
from routes import base , labels_response

app = FastAPI()

app.include_router(base.base_router)
app.include_router(labels_response.response_route)