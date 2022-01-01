from mlprj.datasets import read_cup
from mlprj.model_selection import grid_search_cv, split_train_params
from mlprj.randomized_nn import RandomizedLayer, RandomizedNetwork
from mlprj.losses import MSE, MEE
from mlprj.utility import model_loss


def randomized_nn_model_cup(units):
    nn = RandomizedNetwork(10, RandomizedLayer(units, "sigmoid"), 2)
    nn.compile(loss=MSE())

    return nn

if __name__ == "__main__":

    X, test_x, y, test_y, _= read_cup()

    # set this dictionary to do the grid search
    cup_params = {
        "units": [...],
        "lambda_": [...],
        "p_d":[...], # probability dropout hidden neurons
        "p_dc":[...] # probability dropconnect hidden weights
    }

    cup_best_params = grid_search_cv(randomized_nn_model_cup, (X, y), cup_params, k_folds = 5, path ="cup_randomized", direct=True)
    cup_best_params_other, cup_best_params_training = split_train_params(cup_best_params, direct=True)
    print(cup_best_params_other, cup_best_params_training)

    model = randomized_nn_model_cup(**cup_best_params_other)
    history = model.direct_training((X, y), **cup_best_params_training, verbose=True)

    print(f"train MEE loss: {model_loss(model, MEE(), X, y)}")
    print(f"test MEE loss: {model_loss(model, MEE(), test_x, test_y)}")