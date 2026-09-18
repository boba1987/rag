from fastapi import FastAPI

from app.api.auth import ApiKeyMiddleware
from app.api.query import router


def create_app(retriever=None, generator=None, extractor=None) -> FastAPI:
    application = FastAPI(title="GetVoIP RAG", version="0.1.0")
    application.add_middleware(ApiKeyMiddleware)
    application.state.retriever = retriever
    application.state.generator = generator
    application.state.extractor = extractor
    application.include_router(router)
    return application


app = create_app()
