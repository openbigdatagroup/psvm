# Introduction #

PSVM must be run in linux environment with mpich2-1.0.6p1.

# Installation #

  * Parallel Infrastructure is based on MPI. Firstly mpich2 must be installed before running PSVM. You can download it from http://www.mcs.anl.gov/research/projects/mpich2/.
    * Install mpich2
      * Download mpich2-1.0.6p1.tar.gz
      * Extract it to a path
      * `./configure`
      * `make`
      * `sudo make install`
      * After installing, you will find some binaries and scripts in $PATH. Test by running `mpd` to see if it exists
    * Create password file `~/.mpd.conf` with access mode 600 (rw-------) in home directory. The file should contain a single line `MPD_SECRETWORD=PASSWORD`. Because you may have many  machines, you must do this on each machine.
      * `touch ~/.mpd.conf`
      * `chmod 600 ~/.mpd.conf`
      * `echo "MPD_SECRETWORD=anypassword" > ~/.mpd.conf`
    * Pick one machine as the master and startup mpd(mpi daemon)
      * `mpd --daemon --listenport=55555`
    * Other machines act as slave and must connect to the master
      * `mpd --daemon -h serverHostName -p 55555`
    * Check whether the environment is setup successfully: on master, run `mpdtrace`, you will see all the slaves. If no machines show up, you must check your mpi setup and refer to mpich2 user manual.
  * Download PSVM package and extract and run `make` in its directory. You will see two binaries generated `svm_train`/`svm_predict`. We use mpich2 builtin compiler mpicxx to compile, it is a wrap of g++.
# Prepare datafile #
  * Our datafile is similar to libsvm. Data is stored using a sparse representation, with one element per line. Each line begins with an integer, which is the element's label and which must be either -1 or 1. The label is then followed by a list of features, of the form featureID:featureValue. It is like `<label> <index1>:<value1> <index2>:<value2> ...` We only support binary classification, so the label must be `1/-1`. Feature index must be in increasing order.
  * The testing/validating data format is the same as the training file format. Note in particular that each element must still have a label, although these labels are only used for computing accuracy, precision, and recall statistics. If you don't have labels for the elements, just provide fake labels.
  * Example: Suppose there are two elements, each with two features. The first one is Feature0: 1, Feature1: 2 and belongs to class 1; The second one is Feature0: 100, Feature1: 200 and belongs to class -1. Then the datafile would look like:
```
1  0:1    1:2
-1 0:100  1:200
```
# Do a simple training and testing #
  * Download [splice](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice) and  [splice.t](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice.t) as training and validating sample
  * Train
    * `mkdir /home/$USER/model/`
    * `./svm_train -rank_ratio 0.1  -kernel_type 2 -hyper_parm 1  -gamma 0.01 -model_path /home/$USER/model/  splice`
  * Predict
    * `./svm_predict -model_path /home/$USER/model/  splice.t`
```
========== Predict Accuracy ==========
Accuracy          : 0.864368
Positive Precision: 0.875899
Positive Recall   : 0.861185
Negative Precision: 0.852305
Negative Recall   : 0.867816
```
  * You will also see a file named PredictResult generated in the /home/$USER/model/ folder. This file stores the predicted labels compared to the original labels of each testing element.
# Command-line flags #
  * Training (svm\_train) flags
    * `kernel_type`:
      * 0: normalized-linear
      * 1: normalized-polynomial
      * 2: RBF Gaussian
      * 3: Laplasian
    * `rank_ratio`: approximation ratio between 0 and 1.  Higher values yield higher accuracy at the cost of increased memory usage and training time.  If you are unsure what value to use, we recommend a value of `1/sqrt(n)`, where `n` is the number of training samples.
    * `hyper_parm`: C in SVM. This is the same as libsvm "-c" parameter
    * `gamma`: gamma value if you use RBF kernel. This is the same as libsvm "-g" parameter
    * `poly_degree`: degree if you use normalized-polynomial kernel. This is the same as libsvm "-d" parameter
    * `model_path`: the location to save the training model and checkpoints to.  Be sure that this path is **EMPTY** before training a new model: svm\_train will interpret any of its checkpoints left in this directory as checkpoints for the current model.
    * `failsafe`: If failsafe is set to true, program will periodically write checkpoints to `model_path` and if program fail, it will restart from last checkpoints.
    * `save_interval`: Because PSVM supports failsafe. On every `save_interval` seconds, program will write a checkpoint. If PSVM fails such as machine is down, it will restart from last checkpoint on next execution.
    * `surrogate_gap_threshold`, `feasible_threshold`, `max_iteration`: Because PSVM use Interior Point Method, there needs many iterations. The iteration will stop by checking ((surrogate\_gap < `surrogate_gap_threshold` and primal residual < `feasible_threshold` and dual residual < `feasible_threshold`) or iterations > `max_iteration`). Usually setting them to default will handle most of the cases.
    * `positive_weight`, `negative_weight`: For unbalanced data, we should set a more-than-one weight to one of the class. For example there are 100 positive data and 10 negative data, it is suggested you set negative\_weight to 10.
    * Others: simply run svm\_train to get description for each parameter. They are not frequently used unless you are quite familiar with algorithm details.
  * Predicting (svm\_predict) flags
    * `model_path`: The path of the model which we use to predict.
    * `output_path`: Where to output `PredictResult`.
# Run PSVM parallelly #
  * Firstly, mpich2 must be setup successfully. Type `mpdtrace` on master to check how many machines we have.
  * Binaries(svm\_train, svm\_predict) must be located on a shared path that each computer can access(The path name should be the same on all computers). You should also prepare a model path which all machines can access.
  * On master: run `mpiexec -n numberOfMachines /PATH/svm_train ....`
    * Simple add `mpiexec -n numberOfMachines` before original command line, then the program will run on parallel machines. (The number of machines specified for execution should not exceed the total number of machines, or two processes will run on a single machine and training time will be limited by this slowest machine.)
    * After training, model.0, model.1 such things will be generated in the shared model\_path
  * On master: run `mpiexec -n numberOfMachines /PATH/svm_predict ....`
    * Note: The number of machines used to predict **MUST** be the **SAME** as the number of machines used to train.