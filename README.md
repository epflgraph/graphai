# GraphAI

This package contains projects and services enhancing the [EPFL Graph](https://www.epfl.ch/education/educational-initiatives/cede/campusanalytics/epfl-graph/) project with AI-based utilities.

* [Setup](#setup)
* [API](#api)
* [Documentation](#documentation)

## Setup
GraphAI requires both RabbitMQ (as the message broker backend) and Redis (as the results backend) to be installed. 
To ensure the requirements are met, run:
```
sudo apt-get install redis
sudo apt-get install rabbitmq
```

The audio/video endpoints additionally require `ffmpeg` and `chromaprint` to be installed.

The GraphAI module may then be installed as a regular python package. To install it, simply run
```
pip install -e .
```

Finally, create and populate the ``config`` folder. The code expects ``config`` to contain the following files:

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
username: <es username>
password: <es password>
cafile: <path to es cluster certificate> 
```

## API
The GraphAI module includes an API that leverages the [FastAPI](https://fastapi.tiangolo.com/) package.

### Deployment
To deploy it, first run the ``deploy_celery.sh`` script in the [graphai/api](graphai/api) folder and then run the ``deploy.sh`` script specifying the host. The app will be listening to the port 28800 by default. For more information about the API endpoints, check its own documentation.

### Development
New endpoints can be added either to an existing router or to a new one.

To add an endpoint to an existing router:
1. Create an async function in the corresponding router file (e.g. [graphai/api/routers/video.py](graphai/api/routers/video.py)), decorated with fastapi's decorator specifying the HTTP method and endpoint name.
2. Create also input and output schemas as classes in the corresponding schema file (e.g. [graphai/api/schemas/video.py](graphai/api/schemas/video.py)). These classes should inherit from [pydantic](https://docs.pydantic.dev/)'s ``BaseModel``, and be named by convention like ``CamelCasedEndpointNameRequest`` and either ``CamelCasedEndpointNameResponse`` or ``CamelCasedEndpointNameResponseElem`` with ``CamelCasedEndpointNameResponse = List[CamelCasedEndpointNameResponseElem]``.
3. Specify these classes as input and output schemas in the function definition in the router.
4. Populate the function with the needed logic.

To add an endpoint to a new router:
1. Create an empty schema file (e.g. [graphai/api/schemas/new.py](graphai/api/schemas/new.py)).
2. Create a router file (e.g. [graphai/api/routers/new.py](graphai/api/routers/new.py)), instantiating a fastapi ``APIRouter`` as follows

        router = APIRouter(
            prefix='/new',
            tags=['new'],
            responses={404: {'description': 'Not found'}}
        )
3. Register the router in the fastapi application by adding to the [graphai/api/main.py](graphai/api/main.py) file the lines

        import graphai.api.routers.new as new_router

        [...]

        app.include_router(new_router.router)
4. At this point, the new router is already created. Create and endpoint on the new router by following the instructions above. Endpoints on this router are available under the ``/new`` prefix.

**Note**: New functionalities should be developed in the [graphai/core](graphai/core) submodule, to make them modular and reusable. API endpoints should be limited to the management of input and output and the orchestration of the different tasks, and should generally rely on functions from that submodule for the actual computation of results.

## Documentation
Documentation of the GraphAI python package is available [here](https://epflgraph.github.io/graphai/graphai).

Documentation of the GraphAI API endpoints is available on the ``/docs`` endpoint of the API ([test](http://test-graphai.epfl.ch/docs) and [prod](http://graphai.epfl.ch:28800/docs)).