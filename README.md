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
`chromaprint` (for audio fingerprinting), `tesseract` (for OCR that is used in slide detection), 
and `poppler-utils` (used for splitting a PDF file into images). Install them with
```
sudo apt-get install ffmpeg
sudo apt-get install libchromaprint-tools
sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-script-latn
sudo apt-get install poppler-utils
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

#### Fasttext initialization
The slide detection pipeline requires English and French-language fasttext models in order to work. 
To initialize these two, run the following two lines, replacing `<PATH>` and `<DIM>` with the `path` and `dim` 
values from the `[fasttext]` section of your `config.ini` file:
```
fasttext-reduce --root_dir <PATH> --lang en --dim <DIM>
fasttext-reduce --root_dir <PATH> --lang fr --dim <DIM>
```

### Authentication
From version 0.3.0 onwards, the API uses bearer tokens for authentication. In order for this authentication to work, 
follow these steps:

1. Add an `[auth]` section to the `config.ini` file, as shown in the example config file. The secret key can be generated using `openssl rand -hex 32`. 
Generate your own and do NOT use the secret key included in the example config file!
2. Modify and run the SQL file `init_auth_db.sql`, found in the `queries/auth` folder in the root directory of the repo in 
order to create your users table and add the first user.
   1. Be sure to fill in the details of the user. Using your desired password, you can generate the value for
   `hashed_password` using the function `graphai.api.auth.auth_utils.get_password_hash`, which uses 
   bcrypt encryption and is used by the API itself.
   2. Make sure that the name of the schema matches the one indicated in the `[auth]` section of `config.ini`. These 
   two are set to `auth_graphai` by default.
3. Create further users using the same SQL file, if desired. You can restrict a user's access to certain endpoint groups 
by removing the corresponding scopes from the `scopes` column for that user.

Now, your users will be able to log in and obtain bearer tokens through the `/token` endpoint as follows:
```
curl -X 'POST' 'http://host:28800/token' -H 'accept: application/json' -H 'Content-Type: application/x-www-form-urlencoded' -d 'grant_type=&username=YOURUSERNAME&password=YOURPASSWORD&scope=&client_id=&client_secret='
```

The access token received from this endpoint can then be used as part of the header for requests to other endpoints:
```
-H 'Authorization: Bearer ACCESS_TOKEN_GOES_HERE'
```

### Rate limiting
The API has built-in hooks for rate-limiting users, since many endpoints are resource-intensive and the server could 
get overloaded if too many requests are received. Rate limiting is disabled by default. In order to enable it, follow 
these steps:
1. Add a `[ratelimiting]` section to the `config.ini` file.
2. Add a `limit` variable to this section, which selects your desired rate-limiting schema. Two schemas are available
by default: `unlimited` (which is self-explanatory) and `base`, which uses a predefined set of sensible rate limits 
for each endpoint group (plus one "global" rate limit that applies to the entire API).

If you need finer-grained control over the rate limits, read on.

#### Advanced rate limiting
1. If you would like to use custom rate-limiting schemas, create a JSON file whose structure mimics
`graphai.api.auth.auth_utils.DEFAULT_RATE_LIMITS`, but with your own custom values and custom names for the schemas. 
Then add a `custom_limits` variable to your `config.ini` file and set its value to the absolute path of your custom
JSON. By doing so, you will be able to set the `limit` variable to schema names included in your custom JSON,
in addition to the default values.
    1. Do not name your custom schemas `base` or `unlimited`, as they will be overwritten by the defaults.
2. If you want to override rate-limit values for any given user and any given endpoint group (`global`, `video`, 
`image`, `voice`, or `translation`), add a row to the table `User_Rate_Limits` (whose definition is found in 
the SQL file `init_auth_db.sql`) with your desired `max_requests` and `window_size` values. For example, if you want 
to set the `global` rate limit to 2000/second for the user `admin`, the row would look like this:
```mysql
INSERT INTO `auth_graphai`.`User_Rate_Limits`
(`username`,
`api_path`,
`max_requests`,
`window_size`)
VALUES
('admin',
'global',
2000,
1);
```
   1. Setting either `max_requests` or `window_size` to `NULL` will disable rate limiting for the given user+path. If you 
   want to disable rate limiting entirely for the user `'admin'`, set its `max_requests`/`window_size` values to `NULL` for 
   every single one of the endpoint groups (plus `global`).
## Deployment
To deploy the API, make sure the RabbitMQ and Redis services are running and accessible at the urls provided in the corresponding config file.

### Deploy Celery
To deploy Celery, run the [deploy_celery.sh](graphai/api/main/deploy_celery.sh) script from the [graphai/api/main](graphai/api/main) folder:
```
cd /path/to/project/graphai/api/main
./deploy_celery.sh
```
 
> ℹ️ Celery can run jobs using **threads** or **processes**. Because of the [Global Interpreter Lock (GIL)](https://docs.celeryq.dev/projects/celery-enhancement-proposals/en/latest/glossary.html#term-GIL) in Python, running using threads will mostly reduce your tasks from true parallelism to concurrency (although this is only true when running Python bytecode, as opposed to e.g. C libraries underneath or I/O calls, which *are* truly run in parallel). However, using processes means that you will have one copy of your read-only (potentially large) objects *per process*.
> 
> The `deploy_celery.sh` script launches multiple workers. Some of these use threads and some use processes (prefork). The workers with higher values of the prefetch multiplier are designed for short, high-volume tasks, while a smaller prefetch multiplier is better suited to long-running tasks.
> When writing new tasks and jobs, be mindful of their overall priority and assign them to the appropriate worker (either using the existing queues or by adding a new queue or even a new worker to `celery.common.celery_config` and `deploy_celery.sh`).
> If your tasks do not use any large objects but need to run fast, you can use the `prefork` pool in order to use true parallelism.
> * Launching multiple workers also allows you to set the niceness value of each worker process, thus giving you more control over priorities.
> * For more optimization tips, particularly regarding the `--prefetch-multiplier` flag, see [here](https://docs.celeryq.dev/en/stable/userguide/optimizing.html#optimizing-prefetch-limit).

> ℹ️ The `deploy_celery.sh` script launches workers in detached mode. You can monitor them using the Flower API, which resides at `localhost:5555` by default. In order to terminate them all, use the `cleanup_celery_workers.sh` script. See more [here](https://docs.celeryq.dev/en/stable/userguide/workers.html#stopping-the-worker).

### Deploy the API
Once the Celery workers are running, to deploy the API, run the [deploy.sh](graphai/api/main/deploy.sh) script from the [graphai/api/main](graphai/api/main) folder:
```
cd /path/to/project/graphai/api/main
./deploy.sh
```
The app will be listening on `0.0.0.0:28800` by default. You can change both the host and the port with the `-h` and `-p` options:
```
./deploy.sh -h <host> -p <port>
```

## Documentation
Documentation of the GraphAI python package is available [here](https://epflgraph.github.io/graphai/graphai).

Documentation of the GraphAI API endpoints is available on the ``/docs`` endpoint of the API ([prod](http://graphai.epfl.ch/docs)).

## License

This project is licensed under the [Apache License 2.0](./LICENSE).