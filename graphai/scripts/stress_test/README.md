# Stress testing

This directory contains a `stress_test.py` script, which can be used to stress test the server by sending it a large 
number of requests in parallel. The requests currently used are translation and video retrieval + slide detection. 

To run the script, follow these steps:
1. Install the `graphai-client` package by running: `pip install git@github.com:epflgraph/graphai-client.git`
2. Create a config file containing your GraphAI credentials and the server address. Example:
`
{
  "host": "http://localhost",
  "port": 28800,
  "user": "USER",
  "password": "PASS"
}
`
3. Run the script with `python stress_test.py --config CONFIG_FILE_PATH --requests N_REQUESTS`. If you don't provide the 
`--requests` argument, its default value will be 10, meaning 10 requests will spawn randomly.

