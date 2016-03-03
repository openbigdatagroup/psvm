CC = mpicxx
FLAGS = -O3 -Wall

SRC_DIR = src
OBJ_DIR = obj
FILE = io.cc util.cc timer.cc document.cc kernel.cc model.cc matrix.cc \
	   matrix_manipulation.cc parallel_interface.cc pd_ipm.cc
SRC = $(addprefix $(SRC_DIR), $(FILE))
OBJ = $(addprefix $(OBJ_DIR)/, $(patsubst %.cc, %.o, $(FILE)))

all: svm_train svm_predict

svm_train: $(OBJ_DIR)/svm_train.o $(OBJ)
	$(CC) $^ -o $@

svm_predict: $(OBJ_DIR)/svm_predict.o $(OBJ)
	$(CC) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc $(SRC_DIR)/%.h
	mkdir -p $(OBJ_DIR)
	$(CC) $< -o $@ -c $(FLAGS)

clean:
	rm -rf $(OBJ_DIR) svm_train svm_predict