from mlprj.datasets import read_monk
from mlprj.model_selection import grid_search_cv, split_train_params
from mlprj.utility import model_accuracy
from mlprj.randomized_nn import RandomizedLayer, RandomizedNetwork
from mlprj.losses import MSE


def randomized_nn_model_monk(units):
    nn = RandomizedNetwork(17, RandomizedLayer(units, "sigmoid"), 1)
    nn.compile(loss = MSE())
    
    return nn


if __name__ == "__main__":
    monk_2_X, monk_2_test_x, monk_2_y, monk_2_test_y = read_monk(2)

    monk_2_params = {
        "units": [100, 200, 300, 400, 500],
        "lambda_": [0, 0.01, 0.1, 1, 10, 100],
        "p_d":[0, 0.1, 0.2, 0.3, 0.4, 0.5], # probability dropout hidden neurons
        "p_dc":[0, 0.1, 0.2, 0.3, 0.4, 0.5] # probability dropconnect hidden weights
        }

    monk_2_best_params = grid_search_cv(randomized_nn_model_monk, (monk_2_X, monk_2_y), monk_2_params, k_folds = 5, direct = True, path="monk2_randomized")
    monk_2_best_params_other, monk_2_best_params_training = split_train_params(monk_2_best_params, direct = True)
    print(monk_2_best_params_other, monk_2_best_params_training)

    model = randomized_nn_model_monk(**monk_2_best_params_other)
    error_tr, error_vl = model.direct_training((monk_2_X, monk_2_y), **monk_2_best_params_training, verbose = True)

    print(f"training accuracy: {model_accuracy(model, monk_2_X, monk_2_y, threshold = 0.5)}")
    print(f"test accuracy: {model_accuracy(model, monk_2_test_x, monk_2_test_y, threshold = 0.5)}")