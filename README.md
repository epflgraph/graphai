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

### Configuration file
The project requires a `config.ini` that is used to specify the configuration for the different connections and AI models. An [example-config.ini](example-config.ini) file is provided with some default values. Copy this file to `config.ini` with

```
cp example-config.ini config.ini
```

then edit it with your credentials and preferences.

> ℹ️ The caching additionally requires either the existence of the schema whose name is specified in the file (`cache_graphai` by default) in your database or otherwise the permission to create a new schema.

### Authentication
From version 0.3.0 onwards, the API uses bearer tokens for authentication. In order for this authentication to work, 
follow these steps:

1. Add an `[auth]` section to the `config.ini` file, as shown in the example config file. The secret key can be generated using `openssl rand -hex 32`. 
Generate your own and do NOT use the secret key included in the example config file!
2. Modify and run the SQL file `init_auth_db.sql`, found in the `queries` folder in the root directory of the repo in 
order to create your users table and add the first user.
   1. Be sure to fill in the details of the user. Using your desired password, you can generate the value for
   `hashed_password` using the function `graphai.core.common.auth_utils.get_password_hash`, which uses 
   bcrypt encryption and is used by the API itself.
   2. Make sure that the name of the schema matches the one indicated in the `[auth]` section of `config.ini`. These 
   two are set to `auth_graphai` by default.
3. Create further users using the same SQL file, if desired. You can restrict a user's access to certain endpoint groups 
by removing the corresponding scopes from the `scopes` column for that user.

Now, your users will be able to log in and obtain bearer tokens through the `/token` endpoint, which will grant them 
access to virtually every other endpoint.

## Deployment
To deploy the API, make sure the RabbitMQ and Redis services are running and accessible at the urls provided in the corresponding config file.

### Deploy Celery
To deploy Celery, run the [deploy_celery.sh](graphai/api/deploy_celery.sh) script from the [graphai/api](graphai/api) folder:
```
cd /path/to/project/graphai/api
./deploy_celery.sh
```
 
> ℹ️ Celery can run jobs using **threads** or **processes**. Because of the [Global Interpreter Lock (GIL)](https://docs.celeryq.dev/projects/celery-enhancement-proposals/en/latest/glossary.html#term-GIL) in Python, running using threads will mostly reduce your tasks from true parallelism to concurrency (although this is only true when running Python bytecode, as opposed to e.g. C libraries underneath or I/O calls, which *are* truly run in parallel). However, using processes means that you will have one copy of your read-only (potentially large) objects *per process*.
> 
> The `deploy_celery.sh` script launches three workers, two of which use threads. Out of those two, one is designed to handle time-critical tasks, and another is designed to handle long-running tasks. When writing new tasks and jobs, be mindful of their overall priority and assign them to the appropriate worker (either using the existing queues or by adding a new queue to `core/interfaces/celery_config.py` and `deploy_celery.sh`).
> If your tasks do not use any large objects but need to run fast, you can use the `prefork` pool in order to use true parallelism.
> * Launching multiple workers also allows you to set the niceness value of each worker process, thus giving you more control over priorities.
> * For more optimization tips, particularly regarding the `--prefetch-multiplier` flag, see [here](https://docs.celeryq.dev/en/stable/userguide/optimizing.html#optimizing-prefetch-limit).

> ℹ️ The `deploy_celery.sh` script launches workers in detached mode. You can monitor them using the Flower API, which resides at `localhost:5555` by default. In order to terminate them all, use the `cleanup_celery_workers.sh` script. See more [here](https://docs.celeryq.dev/en/stable/userguide/workers.html#stopping-the-worker).

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