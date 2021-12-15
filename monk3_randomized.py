import matplotlib.pyplot as plt

from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import *
from mlprj.model_selection import *
from mlprj.losses import *
from mlprj.regularizers import *
from mlprj.initializers import *
from mlprj.feed_forward import Network, Layer
from mlprj.utility import model_accuracy
from mlprj.randomized_nn import *


def randomized_nn_model_monk(units):
    nn = RandomizedNetwork(17, [
        RandomizedLayer(units, "relu"),
        RandomizedLayer(1, "linear")
    ])
    
    nn.compile(loss=MSE())
    
    return nn

if __name__ == "__main__":
   
    monk_3_X, monk_3_test_x, monk_3_y, monk_3_test_y = read_monk(3)
    monk_3_params = {
        "units": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "lambda_": [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }

    monk_3_best_params = grid_search_cv(randomized_nn_model_monk, (monk_3_X, monk_3_y), monk_3_params, k_folds=5, direct=True)
    monk_3_best_params_other, monk_3_best_params_training = split_train_params(monk_3_best_params, direct=True)
    print(monk_3_best_params_other, monk_3_best_params_training)

    model = randomized_nn_model_monk(**monk_3_best_params_other)
    error_tr, error_vl = model.direct_training((monk_3_X, monk_3_y), **monk_3_best_params_training, verbose = True)

    print(f"training accuracy: {model_accuracy(model, monk_3_X, monk_3_y, threshold = 0.5)}")
    print(f"test accuracy: {model_accuracy(model, monk_3_test_x, monk_3_test_y, threshold = 0.5)}")