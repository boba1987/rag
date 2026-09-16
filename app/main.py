from fastapi import FastAPI

from app.api.query import router


def create_app(retriever=None, generator=None) -> FastAPI:
    application = FastAPI(title="GetVoIP RAG", version="0.1.0")
    application.state.retriever = retriever
    application.state.generator = generator
    application.include_router(router)
    return application


app = create_app()
