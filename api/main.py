from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.routers import text

# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph AI API",
    description="This API offers several tools related with AI in the context of the EPFL Graph project, "
                "such as automatized concept detection from a given text.",
    version="0.2.0"
)

# Include all routers in the app
app.include_router(text.router)


@app.get("/")
async def redirect_docs():
    return RedirectResponse(url='/docs')
