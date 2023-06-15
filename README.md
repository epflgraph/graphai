# GraphAI

This package contains projects and services enhancing the [EPFL Graph](https://www.epfl.ch/education/educational-initiatives/cede/campusanalytics/epfl-graph/) project with AI-based utilities.

## Contents

* [Overview](#overview)
* [Setup](#setup)
* [Deployment](#deployment)
* [Development](#development)
* [Documentation](#documentation)

## Overview
GraphAI is composed of an API, implemented using the [FastAPI](https://fastapi.tiangolo.com/) package, which makes the different services available. These services, that have different ranges of execution time and desired priority, are run through tasks using [Celery](https://docs.celeryq.dev/en/stable/).

To provide these services, GraphAI connects to a database that contains graph-like data (nodes and edges), the details of which can be specified in its configuration. Additionally, it connects by default to an elasticsearch cluster that contains concepts (Wikipedia pages) in a Mediawiki-like format, although this connection is optional and can be replaced with a request to the Mediawiki API by specifying a parameter in the request.

There are two types of services: *synchronous* and *asynchronous*. The former are usual REST API endpoints: given an input, they perform some operations and return the result. The latter are composed of two endpoints: one that queues a task and immediately returns a task id, and another that, with the task id as input, polls whether the result is available.

## Setup
### Install RabbitMQ and Redis
Aside from the regular dependencies, that are specified in the [pyproject.toml](pyproject.toml) file included in the project, GraphAI relies on Celery, which requires a message broker and a results backend. GraphAI works with [RabbitMQ](https://www.rabbitmq.com/) as the message broker and [Redis](https://redis.io/) as the results backend. In ubuntu, you can install these packages with
```
sudo apt-get install rabbitmq-server
sudo apt-get install redis
```

### Install audio/video processing software
The audio/video endpoints require the packages `ffmpeg` (for general handling of video/audio), 
`chromaprint` (for audio fingerprinting), and `tesseract` (for OCR that is used in slide detection). Install them with
```
sudo apt-get install ffmpeg
sudo apt-get install libchromaprint-tools
sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-script-latn
```

### Install GraphAI python package
> ℹ️ At this point we advise to create and activate an isolated python virtual environment.

The GraphAI package may then be installed as a regular python package. To install it, simply run
```
pip install -e .
```
All python dependencies will be automatically installed.

### Configuration files
The project has a [config](config) folder that is used to specify the configuration for the different connections and AI models. Only the database and Celery files are required, the rest have default fallback mechanisms when files are absent.

#### Database

**db.ini**: Database config file, a file with this information:
```
[DB]
host: <db host>
port: <db port>
user: <db user>
password: <db password>
```

#### Celery

**celery.ini**: Celery config file:
```
[CELERY]
broker_url: <rabbitmq message broker url> (e.g. amqp://guest:guest@localhost:5672//)
result_backend: <redis backend url> (e.g. redis://localhost:6379/0)
```

#### Elasticsearch

**es.ini**: Elasticsearch config file, a file with this information:
```
[ES]
host: <es host>
port: <es port>
username: <es username>
password: <es password>
cafile: <path to es cluster certificate> 
```

#### AI models

**models.ini**: Config file for Whisper model (audio transcription) and Google Vision API (OCR):
```
[WHISPER]
model_type: <whisper model type, default medium>

[GOOGLE]
api_key: <Google API key>
```

#### Caching

**cache.ini**: Config file for results caching:
```
[CACHE]
root: <root directory for the storage of all files (video, audio, image, etc.)>
```

> ℹ️ The caching additionally requires either the existence of the schema `cache_graphai` in your database or otherwise the 
permission to create a new schema.

## Deployment
To deploy the API, make sure the RabbitMQ and Redis services are running and accessible at the urls provided in the corresponding config file.

### Deploy Celery
To deploy Celery, run the [deploy_celery.sh](graphai/api/deploy_celery.sh) script from the [graphai/api](graphai/api) folder:
```
cd /path/to/project/graphai/api
./deploy_celery.sh
```
 
> ℹ️ Celery can run jobs using **threads** or **processes**. Because of the [Global Interpreter Lock (GIL)](https://docs.celeryq.dev/projects/celery-enhancement-proposals/en/latest/glossary.html#term-GIL) in Python, running using threads will mostly reduce your tasks from true parallelism to concurrency (although this is only true when running Python bytecode, as opposed to e.g. C libraries underneath, which *are* truly run in parallel). However, using processes means that you will have one copy of your read-only (potentially large) objects *per process*.
> 
> The `deploy_celery.sh` script launches two workers, both of which use threads, one designed to handle time-critical tasks, and another designed to handle long-running tasks. When writing tasks and jobs, be mindful of their overall priority and assign them to the appropriate worker (either using the existing queues or by adding a new queue to `core/interfaces/celery_config.py` and `deploy_celery.sh`).
> * Launching multiple workers also allows you to set the niceness value of each worker process, thus giving you more control over priorities.
> * For more optimization tips, particularly regarding the `--prefetch-multiplier` flag, see [here](https://docs.celeryq.dev/en/stable/userguide/optimizing.html#optimizing-prefetch-limit).

> ℹ️ The `deploy_celery.sh` script launches workers in detached mode. In order to monitor the status of **all** workers, use the `monitor_celery.sh` script. In order to terminate them all, use the `cleanup_celery_workers.sh` script. See more [here](https://docs.celeryq.dev/en/stable/userguide/workers.html#stopping-the-worker).

### Deploy the API
Once the Celery workers are running, to deploy the API, run the [deploy.sh](graphai/api/deploy.sh) script from the [graphai/api](graphai/api) folder:
```
cd /path/to/project/graphai/api
./deploy.sh
```
The app will be listening on `0.0.0.0:28800` by default. You can change both the host and the port with the `-h` and `-p` options:
```
./deploy.sh -h <host> -p <port>
```

## Development
New endpoints can be added either to an existing router or to a new one.

To add an endpoint to an existing router:
1. Create an async function in the corresponding router file (e.g. [graphai/api/routers/video.py](graphai/api/routers/video.py)), decorated with FastAPI's decorator specifying the HTTP method and endpoint name.
2. Create also input and output schemas as classes in the corresponding schema file (e.g. [graphai/api/schemas/video.py](graphai/api/schemas/video.py)). These classes should inherit from [pydantic](https://docs.pydantic.dev/)'s ``BaseModel``, and be named by convention like ``NewEndpointRequest`` and either ``NewEndpointResponse`` or ``NewEndpointResponseElem`` with ``NewEndpointResponse = List[NewEndpointResponseElem]``.
3. Specify these classes as input and output schemas in the function definition in the router.
4. Populate the function with the needed logic.

To add an endpoint to a new router:
1. Create an empty schema file (e.g. [graphai/api/schemas/new.py](graphai/api/schemas/new.py)).
2. Create a router file (e.g. [graphai/api/routers/new.py](graphai/api/routers/new.py)), instantiating a fastapi ``APIRouter`` as follows
    ```
    router = APIRouter(
        prefix='/new',
        tags=['new'],
        responses={404: {'description': 'Not found'}}
    )
    ```
3. Register the router in the fastapi application by adding to the [graphai/api/main.py](graphai/api/main.py) file the lines
    ```
    import graphai.api.routers.new as new_router

    [...]

    app.include_router(new_router.router)
    ```
4. At this point, the new router is already created. Create and endpoint on the new router by following the instructions above. Endpoints on this router are available under the ``/new`` prefix.

> ℹ️ New functionalities should be developed in the [graphai/core](graphai/core) submodule, to make them modular and reusable. API endpoints should be limited to the management of input and output and the orchestration of the different Celery tasks, and should generally rely on functions from that submodule for the actual computation of results.

## Documentation
Documentation of the GraphAI python package is available [here](https://epflgraph.github.io/graphai/graphai).

Documentation of the GraphAI API endpoints is available on the ``/docs`` endpoint of the API ([test](http://test-graphai.epfl.ch/docs) and [prod](http://graphai.epfl.ch/docs)).