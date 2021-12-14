import matplotlib.pyplot as plt

from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import *
from mlprj.model_selection import *
from mlprj.losses import *
from mlprj.regularizers import *
from mlprj.initializers import *
from mlprj.feed_forward import Network, Layer
from mlprj.randomized_nn import *


def monk_2_build_model(learning_rate, alpha):
    nn = Network(17, [
        Layer(5, "relu", UniformInitializer(min_value=-0.1, max_value=0.1)),
        Layer(1, "sigmoid", UniformInitializer(min_value=-0.1, max_value=0.1))
    ])

    nn.compile(loss=MSE(), regularizer=L2Regularizer(0), optimizer=StochasticGradientDescent(learning_rate, alpha))

    return nn


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
        "learning_rate": [0.7, 0.8],
        "alpha": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "epochs": 500,
        "batch_size": "full"
    }

    monk_2_best_params = grid_search_cv(monk_2_build_model, (monk_2_X, monk_2_y), monk_2_params)
    monk_2_best_params_other, monk_2_best_params_training = split_train_params(monk_2_best_params)
    print(monk_2_best_params_other, monk_2_best_params_training)

    model = monk_2_build_model(**monk_2_best_params_other)
    history = model.training((monk_2_X, monk_2_y), epochs=500, batch_size="full", verbose = True)

    plt.plot(history["loss_tr"])
    plt.plot(history["loss_vl"])
    plt.show()

    monk_2_params = {
        "units": [2750, 3000, 3250],
        "lambda_": [0.01, 0.1, 1],
        }

    monk_2_best_params = grid_search_cv(randomized_nn_model_monk, (monk_2_X, monk_2_y), monk_2_params, k_folds = 5, direct = True)
    monk_2_best_params_other, monk_2_best_params_training = split_train_params(monk_2_best_params, direct = True)
    print(monk_2_best_params_other, monk_2_best_params_training)

    model = randomized_nn_model_monk(300)
    error_tr, error_vl = model.direct_training((monk_2_X, monk_2_y), lambda_ = 0.1, verbose = True)