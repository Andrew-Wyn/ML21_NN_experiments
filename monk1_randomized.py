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
        RandomizedLayer(1, "linear"),
        ])
    nn.compile(loss = MSE())
    
    return nn


if __name__ == "__main__":

    # Monk-1 datas
    monk_1_X, monk_1_test_x, monk_1_y, monk_1_test_y = read_monk(1)

    monk_1_params = {
        "units": [2750, 3000, 3250],
        "lambda_": [0.01, 0.1, 1],
        }

    monk_1_best_params = grid_search_cv(randomized_nn_model_monk, (monk_1_X, monk_1_y), monk_1_params, k_folds = 5, direct = True, path="monk1_randomized.csv")
    monk_1_best_params_other, monk_1_best_params_training = split_train_params(monk_1_best_params, direct = True)
    print(monk_1_best_params_other, monk_1_best_params_training)

    model = randomized_nn_model_monk(**monk_1_best_params_other)
    error_tr, error_vl = model.direct_training((monk_1_X, monk_1_y), **monk_1_best_params_training, verbose = True)

    print(f"training accuracy: {model_accuracy(model, monk_1_X, monk_1_y, threshold = 0.5)}")
    print(f"test accuracy: {model_accuracy(model, monk_1_test_x, monk_1_test_y, threshold = 0.5)}")