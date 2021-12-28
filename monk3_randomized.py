from mlprj.datasets import read_monk
from mlprj.model_selection import grid_search_cv, split_train_params
from mlprj.utility import model_accuracy
from mlprj.randomized_nn import RandomizedLayer, RandomizedNetwork
from mlprj.losses import MSE


def randomized_nn_model_monk(units):
    nn = RandomizedNetwork(17, RandomizedLayer(units, "sigmoid"), 1)
    nn.compile(loss=MSE())
    
    return nn


if __name__ == "__main__":
   
    monk_3_X, monk_3_test_x, monk_3_y, monk_3_test_y = read_monk(3)
    monk_3_params = {
        "units": [2750, 3000, 3250],
        "lambda_": [0.01, 0.1, 1],
        "p_d":[0.1, 0.2], # probability dropout hidden neurons
        "p_dc":[0.1, 0.2] # probability dropconnect hidden weights
        }

    monk_3_best_params = grid_search_cv(randomized_nn_model_monk, (monk_3_X, monk_3_y), monk_3_params, k_folds=5, direct=True, path="monk3_randomized")
    monk_3_best_params_other, monk_3_best_params_training = split_train_params(monk_3_best_params, direct=True)
    print(monk_3_best_params_other, monk_3_best_params_training)

    model = randomized_nn_model_monk(**monk_3_best_params_other)
    history = model.direct_training((monk_3_X, monk_3_y), **monk_3_best_params_training, verbose = True)

    print(f"training accuracy: {model_accuracy(model, monk_3_X, monk_3_y, threshold = 0.5)}")
    print(f"test accuracy: {model_accuracy(model, monk_3_test_x, monk_3_test_y, threshold = 0.5)}")