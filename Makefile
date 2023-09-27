PYTHON=python3

all: split train predict

split:
	@(cd split ; $(PYTHON) split.py)

train:
	@(cd train ; $(PYTHON) train.py)

predict:
	@(cd predict ; $(PYTHON) predict.py)

clean:
	rm -rf model.pkl
	rm -rf */*/__pycache__
	rm -rf datasets/train.csv
	rm -rf datasets/validation.csv

.PHONY: all split train predict cd_split cd_train cd_predict
