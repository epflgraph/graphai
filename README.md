# GraphAI

This module contains projects and services enhancing the [EPFL Graph](https://www.epfl.ch/education/educational-initiatives/cede/campusanalytics/epfl-graph/) project with AI-based utilities.

* [Setup](#setup)
* [Concept Detection](#concept_detection)
* [Wikitext](#wikitext)

## Setup
The GraphAI module may be installed as a regular python package. To install it, simply run
```
pip install -r requirements.txt
pip install -e .
```

Finally, create and populate the ``data`` and ``config`` folders. The code expects ``data`` to contain the following files:

* **predecessors.json**: JSON file containing a dictionary whose keys are Wikipedia page ids and whose values are the list of predecessors of that page in the concepts graph.
* **successors.json**: JSON file containing a dictionary whose keys are Wikipedia page ids and whose values are the list of successors of that page in the concepts graph.

And it expects ``config`` to contain the following files:

* **db.ini**: Database config file, a file with this information:
```
[DB]
host: <db host>
port: <db port>
user: <db user>
password: <db password>
```

* **es.ini**: Elasticsearch config file, a file with this information:
```
[ES]
host: <es host>
port: <es port>
index: <es index>
```

## Concept detection
The concept detection submodule contains several tools to automate concept detection and extraction in the form of Wikipedia pages.

### Text wikification
The main tool available is text wikification, namely the extraction from raw text of a list of Wikipedia pages associated to keywords in the text which are relevant, together with several scores measuring this relevance.

### Wikimarkup stripper
Another tool which is available is to remove wikimarkup from a text. Its implementation relies on the [mwparserfromhell](https://github.com/earwig/mwparserfromhell) python package, but responds to the custom needs of the EPFL graph project.

### FastAPI app
In order to make the services in the concept detection submodule easily available from anywhere, it includes a [FastAPI](https://fastapi.tiangolo.com/) app. To deploy it, run any of the ``deploy-*.sh`` scripts in [concept_detection/api](concept_detection/api) depending on your environment. The app will be listening to the port 28800 by default. For more information about the API, check its own documentation.
