# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    Makefile                                          :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 12:56:40 by cmariot          #+#    #+#              #
#    Updated: 2023/09/27 12:57:32 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

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

.PHONY: all split train predict clean
