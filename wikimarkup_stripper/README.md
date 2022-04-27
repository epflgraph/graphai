# Wikimarkup stripper

## Introduction
This project provides a tool to remove wiki markup from a text. Its implementation relies on the [mwparserfromhell](https://github.com/earwig/mwparserfromhell) python package, but responds to the custom needs of the EPFL graph project.

## Setup
The wikimarkup stripper may be used in two ways: as a standalone python package or as a [FastAPI](https://fastapi.tiangolo.com/) app.

### Installation and usage as a python package
The simplest way of using the wikimarkup stripper is to install it as a python package. To do that, simply install its dependencies with:

`pip install -r requirements.txt`

Then install the package with:

`pip install .`

Finally, just import the strip function and use it as follows:

```
from wikimarkup_stripper.src.models.stripper import strip

markup_code = "Wikimarkup [Text] to be <b>stripped</b>"
stripped_code = strip(markup_code)
print(stripped_code)    # Prints "Wikimarkup Text to be stripped"
```

### Installation and usage as a FastAPI app
To setup the wikimarkup stripper as a FastAPI app, first install the app requirements with:

`pip install -r requirements-app.txt`

Then deploy the app with

`./deploy-test.sh 0.0.0.0`

by specifying the host of your choice. The app will be listening to the port 30010 by default, and will accept POST requests to the `/strip` entry point. For instance, a call with payload

`{"markup_code": "Wikimarkup [Text] to be <b>stripped</b>"}`

will return the following response:

`{"stripped_code": "Wikimarkup Text to be stripped"}`