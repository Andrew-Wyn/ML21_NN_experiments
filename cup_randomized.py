import matplotlib.pyplot as plt

from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import *
from mlprj.model_selection import *
from mlprj.losses import *
from mlprj.regularizers import *
from mlprj.initializers import *
from mlprj.feed_forward import Network, Layer
from mlprj.randomized_nn import *
from sklearn.model_selection import train_test_split


def randomized_nn_model_cup(units):
    nn = RandomizedNetwork(10, [
        RandomizedLayer(units, "relu"),
        RandomizedLayer(2, "linear")
    ])
    
    nn.compile(loss=MSE())
    
    return nn

if __name__ == "__main__":

    cup_X, cup_y= read_cup()
    cup_train_x, cup_test_x, cup_train_y, cup_test_y = train_test_split(cup_X, cup_y, test_size=0.10, random_state=42)

    cup_params = {
        "units": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "lambda_": [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }

    cup_best_params = grid_search_cv(randomized_nn_model_cup, (cup_train_x, cup_train_y), cup_params, k_folds = 5)
    cup_best_params_other, cup_best_params_training = split_train_params(cup_best_params, direct = False)
    print(cup_best_params_other, cup_best_params_training)

    model = randomized_nn_model_cup(**cup_best_params_other)
    error_tr, error_vl = model.direct_training(((cup_train_x, cup_train_y), (cup_test_x, cup_test_y)), **cup_best_params_training, verbose = True)

