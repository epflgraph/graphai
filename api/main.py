from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.routers import ontology
from api.routers import text
from api.routers import video

# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph AI API",
    description="This API offers several tools related with AI in the context of the EPFL Graph project, "
                "such as automatized concept detection from a given text.",
    version="0.2.0"
)

# Include all routers in the app
app.include_router(ontology.router)
app.include_router(text.router)
app.include_router(video.router)


# Root endpoint redirects to docs
@app.get("/")
async def redirect_docs():
    return RedirectResponse(url='/docs')
