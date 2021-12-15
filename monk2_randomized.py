import matplotlib.pyplot as plt

from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import *
from mlprj.model_selection import *
from mlprj.losses import *
from mlprj.regularizers import *
from mlprj.initializers import *
from mlprj.feed_forward import Network, Layer
from mlprj.randomized_nn import *


def randomized_nn_model_monk(units):
    nn = RandomizedNetwork(17, [
        RandomizedLayer(units, "relu"),
        RandomizedLayer(1, "linear"),
        ])
    nn.compile(loss = MSE())
    
    return nn

if __name__ == "__main__":
    monk_2_X, monk_2_test_x, monk_2_y, monk_2_test_y = read_monk(2)

    monk_2_params = {
        "units": [2750, 3000, 3250],
        "lambda_": [0.01, 0.1, 1],
        }

    monk_2_best_params = grid_search_cv(randomized_nn_model_monk, (monk_2_X, monk_2_y), monk_2_params, k_folds = 5, direct = True)
    monk_2_best_params_other, monk_2_best_params_training = split_train_params(monk_2_best_params, direct = True)
    print(monk_2_best_params_other, monk_2_best_params_training)

    model = randomized_nn_model_monk(**monk_2_best_params_other)
    error_tr, error_vl = model.direct_training((monk_2_X, monk_2_y), **monk_2_best_params_training, verbose = True)    