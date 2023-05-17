# GraphAI tests
GraphAI testing is done through the [pytest](https://docs.pytest.org) package. Creating tests for functions in the GraphAI package is slightly more involved than usual due to the integration with FastAPI and Celery. For some tests, Celery needs to be running in the background. For other tests, we instead patch the Celery tasks so that they are not actually invoked. For some more tests, we need to mock the FastAPI application so that we can call the endpoints directly without performing an HTTP request.

## Overview
All tests are placed in the [tests](tests) folder, which contains the following files:
* Test files that are prefixed by `test_` (e.g. `test_text.py` or `test_video.py`).
* A `fixtures` folder containing the necessary pytest fixtures to be used in the tests.
* A `conftest.py` file that imports fixtures into the tests.

## Tests structure
In order to keep tests organised, we follow some conventions in their naming and structure.
* There is a test file per router (e.g. `test_text.py`).
* Each test file contains tests for all its endpoints, sequentially and with a clear separation between endpoints.
* For each endpoint, we also test each of the tasks it orchestrates. However, If a task is used in several endpoints, we only test it once.
* For each endpoint, there are three types of tests:
  * Mock task tests: We simulate a call to a Celery task and check that it is called.
  * Run task tests: We perform a call to a Celery task and check that the returned result is correct.
  * Integration tests: We simulate a request to the API endpoint and check that the response is correct. These tests need Celery running in the background.
* The naming convention of the test functions is the following:
  * Mock task tests: `test__<path_to_endpoint>__<task_name>__mock_task`.
  * Run task tests: `test__<path_to_endpoint>__<task_name>__run_task`.
  * Integration tests: `test__<path_to_endpoint>__integration`.

For instance, if a new endpoint `/text/new` is created and uses tasks `split`, `process` and `combine`, then the following test functions should be implemented in the `test_text.py` file:
* `test__text_new__split__mock_task`
* `test__text_new__process__mock_task`
* `test__text_new__combine__mock_task`


* `test__text_new__split__run_task`
* `test__text_new__process__run_task`
* `test__text_new__combine__run_task`


* `test__text_new__integration`

## Run
To run the tests, simply run one of the following:
```
# Run all tests
pytest

# Run all tests in a module
pytest test_text.py

# Run all tests matching an expression
pytest -k "text_new and not integration"
```

Additionally, all tests are run when a pull request is created on the GitHub repository (or when a commit is pushed to a branch with an open pull request). This is done to detect errors *before* merging to the master branch. Should any tests fail, please check and either fix the changed code or update the associated tests.