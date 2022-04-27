import logging
from fastapi import FastAPI
from wikimarkup_stripper.schemas import StripData
from wikimarkup_stripper.stripper import strip

# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph - Wiki Markup Stripper",
    description="This API allows users to clean code containing wiki markup by extracting the text.",
    version="0.1.0"
)

# Get uvicorn logger so we can write on it
logger = logging.getLogger('uvicorn.error')


@app.post('/strip')
async def keywords(data: StripData):
    # Get markup code from parameter
    markup_code = data.markup_code

    # Strip markup code
    text = strip(markup_code)

    return {'stripped_code': text}
