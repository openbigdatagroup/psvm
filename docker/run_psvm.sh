#!/bin/bash
if [ $# != 3 ] ; then 
echo "USAGE: $0 NODE_NUM DATA_NAME RANK_RATIO (example: ./run_psvm.sh 3 splice 0.1)" 
exit 1; 
fi 
NODE_NUM=$1
if [ $NODE_NUM -le 0 ]; then  NODE_NUM=1; fi
DATA_NAME=$2
RANK_RATIO=$3
echo "Building..."
docker exec psvm-master bash -c "cd /root/psvm && make clean && make"

echo "Training..."
docker exec psvm-master bash -c "mkdir -p models/$DATA_NAME && time mpiexec -f ./docker/hosts -n $NODE_NUM ./svm_train -rank_ratio $RANK_RATIO -kernel_type 2 -hyper_parm 1 -gamma 0.01 -model_path ./models/$DATA_NAME ./data/$DATA_NAME"
echo "Finished training."

echo "Testing..."
docker exec psvm-master bash -c "time mpiexec -f ./docker/hosts -n $NODE_NUM ./svm_predict -model_path ./models/$DATA_NAME ./data/${DATA_NAME}.t"
echo "Finished testing."
