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

    # Monk-1 datas
    monk_1_X, monk_1_test_x, monk_1_y, monk_1_test_y = read_monk(1)

    monk_1_params = {
        "units": [200, 300, 500, 600, 700],
        "lambda_": [0.01, 0.1, 0.4, 0.5],
        "p_d":[0.1, 0.2, 0.4], # probability dropout hidden neurons
        "p_dc":[0.1, 0.2, 0.4] # probability dropconnect hidden weights
        }

    monk_1_best_params = grid_search_cv(randomized_nn_model_monk, (monk_1_X, monk_1_y), monk_1_params, k_folds = 5, direct = True, path="monk1_randomized")
    monk_1_best_params_other, monk_1_best_params_training = split_train_params(monk_1_best_params, direct = True)
    print(monk_1_best_params_other, monk_1_best_params_training)

    model = randomized_nn_model_monk(**monk_1_best_params_other)
    error_tr, error_vl = model.direct_training((monk_1_X, monk_1_y), **monk_1_best_params_training, verbose = True)

    print(f"training accuracy: {model_accuracy(model, monk_1_X, monk_1_y, threshold = 0.5)}")
    print(f"test accuracy: {model_accuracy(model, monk_1_test_x, monk_1_test_y, threshold = 0.5)}")