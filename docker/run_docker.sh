#!/bin/bash
if [ $# != 1 ] ; then
echo "Usage: $0 NODE_NUM (example: ./run_docker.sh 3)"
exit 1;
fi

if [[ `docker images -q psvm:latest 2> /dev/null` == "" ]]; then
  echo "Docker image psvm does not exist, build it first..."
  docker build -t psvm .
fi

NODE_NUM=$1
if [ $NODE_NUM -le 0 ]; then  NODE_NUM=1; fi
echo "Setting up $NODE_NUM node"

echo "Run dockers and collect ips..."
# at least setting up 1 node called master
docker run -v $(readlink -f ..):/root/psvm -d -h master --name psvm-master psvm
docker inspect --format '{{ .NetworkSettings.IPAddress }}' psvm-master > hosts
for((i=2; i<=$NODE_NUM; i++)); do
  docker run -v $(readlink -f ..):/root/psvm -d --link=psvm-master:master --name psvm-node-$i psvm
  docker inspect --format '{{ .NetworkSettings.IPAddress }}' psvm-node-$i >> hosts
done
