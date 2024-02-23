# GraphAI queries

This directory contains a collection of queries that are used to configure the API's database 
for its basic operation or to make modifications to various tables that are used by the API. 
All of the queries in this directory are templates and should be filled with your own parameters 
before being executed.

Here is a list of the subfolders and the queries they contain:
* **auth**: This subfolder contains the query that initializes the authentication schema and 
the `Users` table within it, which the API uses for its OAuth2.0 authentication (required for 
versions >= 0.6.0).
* **ontology_modification**: This subfolder contains queries for moving a cluster, splitting 
a cluster, and adding a new concept to the ontology either to an existing cluster or in a new one.