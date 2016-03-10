#!/bin/bash
if [ $# != 1 ] ; then
echo "Usage: $0 NODE_NUM (example: ./clean_docker.sh 3)"
exit 1;
fi

NODE_NUM=$1
if [ $NODE_NUM -le 0 ]; then  NODE_NUM=1; fi
if [ ! $NODE_NUM ]; then  NODE_NUM=1; fi
echo "Stop and remove containers..."
docker stop psvm-master
docker rm psvm-master
for((i=2; i<=$NODE_NUM; i++)); do
  docker stop psvm-node-$i
  docker rm psvm-node-$i
done
echo "Finished."
