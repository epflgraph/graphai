# Contributing to GraphAI

GraphAI is an open-source project. Here is a short guide on how you can contribute to the project.

## Development

Additional development by third parties is through fork+pull request. Direct access to the repository is restricted to the EPFLGraph team.

**Note: Forked contributions must NOT modify the GitHub workflow files. Any such modifications will result in the pull request being denied.**

### Adding new functionalities

The `graphai.core` and `graphai.celery` submodules are where functionalities live. Here is how the logic is divided between the two:

#### GraphAI Core

This submodule contains all the logic of any algorithm being used by an endpoint, aside from execution orchestration.

#### GraphAI Celery

This submodule uses the functions defined in `graphai.core` and orchestrates them in Celery. Each group (e.g. `image`, `ontology`) is divided into two scripts:
* `tasks.py`: This is where Celery tasks are located. Each `task` should ideally be one single line, calling a function from `graphai.core`. Shared objects are managed at this layer.
* `jobs.py`: This is where chains of celery tasks are orchestrated by `job` functions. A `job` function may have one single chain, or a few possible chains that are built according to the input. 

### Adding a new endpoint
New endpoints can be added either to an existing router or to a new one.

To add an endpoint to an existing router:
1. Create an async function in the corresponding router file (e.g. [graphai/api/video/router.py](graphai/api/video/router.py)), decorated with FastAPI's decorator specifying the HTTP method and endpoint name.
2. Create also input and output schemas as classes in the corresponding schema file (e.g. [graphai/api/video/schemas.py](graphai/api/video/schemas.py)). These classes should inherit from [pydantic](https://docs.pydantic.dev/)'s ``BaseModel``, and be named by convention like ``NewEndpointRequest`` and either ``NewEndpointResponse`` or ``NewEndpointResponseElem`` with ``NewEndpointResponse = List[NewEndpointResponseElem]``.
3. Specify these classes as input and output schemas in the function definition in the router.
4. Populate the function with the needed logic.

To add an endpoint to a new router:
1. Create an empty subpackage in the `api` subpackage. (e.g. `graphai/api/new`). Be sure to create an init file.
2. Create a schemas file: `graphai/api/new/schemas.py`, and add the Pydantic schemas there.
3. Create a router file: `graphai/api/new/router.py`, instantiating a fastapi ``APIRouter`` as follows
    ```Python
    router = APIRouter(
        prefix='/new',
        tags=['new'],
        responses={404: {'description': 'Not found'}},
        dependencies=[Security(get_current_active_user, scopes=['newscope'])]
    )
    ```
   Bear in mind that if you use a custom scope for the API router's authentication (like above with `'newscope'`), the value needs to be added to the comma-separated `scopes` field for the target users in your auth_graphai.Users table.
4. Register the router in the fastapi application by adding to the [graphai/api/main/main.py](graphai/api/main/main.py) file the lines
    ```Python
    import graphai.api.new.router as new_router

    [...]

    authenticated_router.include_router(new_router.router)
    ```
   * If your endpoint does not require authentication, you can add it to the `unauthenticated_router` instead, and in this case you can skip the `dependencies` argument in the router's creation in the previous step.
5. At this point, the new router is already created. Create and endpoint on the new router by following the instructions above. Endpoints on this router are available under the ``/new`` prefix. Now each new endpoint can call its relevant `job` function from `graphai.celery.new.jobs`.

> ℹ️ API endpoints should be limited to the management of input and output, followed by calling the relevant Celery job.