
Fix later: The timestamp field need to be handled properly


Need to setup a docker network

setup a test container to run a agent which has two modules
- packet capture
- send captured data to redis streams kind of thing

have a redis setup where agent sends the data to redis and backend pulls out the data from the redis

Analyze what kind of data is used to process for now in backend (is it from raw packets data or from csv data)

after analyzing that, use that format of data for fetching from agent and pulls out the data from redis and batching to give the data to inference input

then make update the results out on api streaming the real-time capturing inference

