CC=mpicxx

# C-Compiler flags
CFLAGS=-O3 -Wall

# linker
LD=mpicxx
LFLAGS=-O3 -Wall

all: svm_train svm_predict

clean:
	rm -f *.o
	rm -f svm_train svm_predict

io.o: io.cc io.h
	$(CC) -c $(CFLAGS)  io.cc -o io.o

util.o: util.cc util.h
	$(CC) -c $(CFLAGS)  util.cc -o util.o

timer.o: timer.cc timer.h
	$(CC) -c $(CFLAGS)  timer.cc -o timer.o

document.o: document.cc document.h
	$(CC) -c $(CFLAGS)  document.cc -o document.o

kernel.o: kernel.cc kernel.h
	$(CC) -c $(CFLAGS)  kernel.cc -o kernel.o

model.o: model.cc model.h
	$(CC) -c $(CFLAGS)  model.cc -o model.o

matrix.o: matrix.cc matrix.h
	$(CC) -c $(CFLAGS)  matrix.cc -o matrix.o

matrix_manipulation.o: matrix_manipulation.cc matrix_manipulation.h
	$(CC) -c $(CFLAGS)  matrix_manipulation.cc -o matrix_manipulation.o

parallel_interface.o: parallel_interface.cc parallel_interface.h
	$(CC) -c $(CFLAGS)  parallel_interface.cc -o parallel_interface.o

pd_ipm.o: pd_ipm.cc pd_ipm.h
	$(CC) -c $(CFLAGS)  pd_ipm.cc -o pd_ipm.o

svm_train.o: svm_train.cc
	$(CC) -c $(CFLAGS)  svm_train.cc -o svm_train.o

svm_predict.o: svm_predict.cc svm_predict.h
	$(CC) -c $(CFLAGS)  svm_predict.cc -o svm_predict.o


svm_train: timer.o io.o parallel_interface.o util.o model.o document.o matrix.o kernel.o matrix_manipulation.o pd_ipm.o svm_train.o
	$(LD) $(LFLAGS) timer.o io.o parallel_interface.o util.o document.o model.o matrix.o kernel.o matrix_manipulation.o pd_ipm.o svm_train.o -o svm_train

svm_predict: svm_predict.o model.o document.o parallel_interface.o
	$(LD) $(LFLAGS) timer.o io.o util.o document.o model.o kernel.o parallel_interface.o svm_predict.o -o svm_predict
