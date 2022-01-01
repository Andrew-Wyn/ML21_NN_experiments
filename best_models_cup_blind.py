import csv

import matplotlib.pyplot as plt

from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import read_cup, read_cup_blind_test
from mlprj.feed_forward import Network, Layer
from mlprj.randomized_nn import RandomizedNetwork, RandomizedLayer

from mlprj.losses import *
from mlprj.regularizers import *
from mlprj.initializers import *
from mlprj.ensemble import *
from mlprj.utility import model_loss


def cup_build_model_1(learning_rate, alpha, lambdareg = 0, decay = None):
    nn = Network(10, [ 
        Layer(50, "relu", GaussianInitializer()),
        Layer(20, "sigmoid", GaussianInitializer()),
        Layer(10, "relu", GaussianInitializer()),
        Layer(2, "linear", GaussianInitializer())
      ])

    nn.compile(loss=MEE(), regularizer=L2Regularizer(lambdareg), optimizer=StochasticGradientDescent(learning_rate, alpha, decay))
    return nn


def cup_build_model_2(learning_rate, alpha, lambdareg = 0, decay = None):
    nn = Network(10, [ 
        Layer(30, "relu", GaussianInitializer()),
        Layer(10, "sigmoid", GaussianInitializer()),
        Layer(10, "relu", GaussianInitializer()),
        Layer(2, "linear", GaussianInitializer())
      ])

    nn.compile(loss=MEE(), regularizer=L2Regularizer(lambdareg), optimizer=StochasticGradientDescent(learning_rate, alpha, decay))
    return nn


def cup_build_model_3(learning_rate, alpha, lambdareg = 0, decay = None):
    nn = Network(10, [ 
        Layer(20, "sigmoid", GaussianInitializer()),
        Layer(10, "relu", GaussianInitializer()),
        Layer(2, "linear", GaussianInitializer())
      ])

    nn.compile(loss=MEE(), regularizer=L2Regularizer(lambdareg), optimizer=StochasticGradientDescent(learning_rate, alpha, decay))
    return nn


def cup_build_model_rand(units):
    nn = RandomizedNetwork(10, RandomizedLayer(units, "sigmoid"), 2)
    nn.compile(loss=MEE())
    return nn


if __name__ == "__main__":

  X, test_x, y, test_y, preprocesser = read_cup()

  blind_test_x = read_cup_blind_test(preprocesser)

  print("started...")

  epochs = 500

  best_model_1_params_build = {
    "learning_rate": 0.15,
    "alpha": 0.125,
    "lambdareg": 0.00005
  }
  
  model1 = cup_build_model_1(**best_model_1_params_build)
  history1 = model1.training((X, y), (test_x, test_y), epochs, batch_size = 32, early_stopping = 20)
  print("Model 1 trained")

  best_model_1_decay_params_build = {
    "learning_rate": 0.15,
    "alpha": 0.225,
    "lambdareg": 0.0001,
    "decay": (250, 0.005)
  }

  model2 = cup_build_model_1(**best_model_1_decay_params_build)
  history2 = model2.training((X, y), (test_x, test_y), epochs, batch_size = 64, early_stopping = 20)
  print("Model 2 trained")

  best_model_2_params_build = {
    "learning_rate": 0.075,
    "alpha": 0.23,
    "lambdareg": 0.00003
  }

  model3 = cup_build_model_2(**best_model_2_params_build)
  history3 = model3.training((X, y), (test_x, test_y), epochs, batch_size = 32, early_stopping = 20)
  print("Model 3 trained")

  best_model_2_decay_params_build = {
    "learning_rate": 0.4,
    "alpha": 0.1,
    "lambdareg": 0.000075,
    "decay": (250, 0.01)
  }


  model4 = cup_build_model_2(**best_model_2_decay_params_build)
  history4 = model4.training((X, y), (test_x, test_y), epochs, batch_size = 32, early_stopping = 20)
  print("Model 4 trained")

  best_model_3_params_build = {
    "learning_rate": 0.2,
    "alpha": 0.175,
    "lambdareg": 0.000025
  }

  model5 = cup_build_model_3(**best_model_3_params_build)
  history5 = model5.training((X, y),(test_x, test_y), epochs, batch_size = 32, early_stopping = 20)
  print("Model 5 trained")

  best_model_3_decay_params_build = {
    "learning_rate": 0.4,
    "alpha": 0.225,
    "lambdareg": 0.00005,
    "decay": (250, 0.01)
  }

  model6 = cup_build_model_3(**best_model_3_decay_params_build)
  history6 = model6.training((X, y), (test_x, test_y), epochs, batch_size = 64, early_stopping = 20)
  print("Model 6 trained")

  model7 = cup_build_model_rand(1000)
  history7 = model7.direct_training((X,y), (test_x, test_y), lambda_=10, p_d=0, p_dc=0)
  print("Model 7 trained")

  plt.plot(history1["loss_tr"], label = "training error")
  plt.plot(history1["loss_vl"], linestyle = "dashed",  label = "internal test set error", color = "r")
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Loss(MEE)', fontsize=14)
  plt.legend()
  plt.show()

  plt.plot(history2["loss_tr"], label = "training error")
  plt.plot(history2["loss_vl"], linestyle = "dashed",  label = "internal test set error", color = "r")
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Loss(MEE)', fontsize=14)
  plt.legend()
  plt.show()

  plt.plot(history3["loss_tr"], label = "training error")
  plt.plot(history3["loss_vl"], linestyle = "dashed",  label = "internal test set error", color = "r")
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Loss(MEE)', fontsize=14)
  plt.legend()
  plt.show()

  plt.plot(history4["loss_tr"], label = "training error")
  plt.plot(history4["loss_vl"], linestyle = "dashed",  label = "internal test set error", color = "r")
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Loss(MEE)', fontsize=14)
  plt.legend()
  plt.show()

  plt.plot(history5["loss_tr"], label = "training error")
  plt.plot(history5["loss_vl"], linestyle = "dashed",  label = "internal test set error", color = "r")
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Loss(MEE)', fontsize=14)
  plt.legend()
  plt.show()

  plt.plot(history6["loss_tr"], label = "training error")
  plt.plot(history6["loss_vl"], linestyle = "dashed",  label = "internal test set error", color = "r")
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Loss(MEE)', fontsize=14)
  plt.legend()
  plt.show()

  tr_results = [history1["loss_tr"][-1], history2["loss_tr"][-1], history3["loss_tr"][-1], history4["loss_tr"][-1], history5["loss_tr"][-1], history6["loss_tr"][-1], history7["loss_tr"]]
  ts_results = [history1["loss_vl"][-1], history2["loss_vl"][-1], history3["loss_vl"][-1], history4["loss_vl"][-1], history5["loss_vl"][-1], history6["loss_vl"][-1], history7["loss_vl"]]

  print("training resuls : ", tr_results)
  print("test resuls : ", ts_results)

  mean_tr_error = np.mean(np.array(tr_results))
  mean_ts_error = np.mean(np.array(ts_results))

  print(f"L'errore medio nel training set è: {mean_tr_error}")
  print(f"L'errore medio nell' internal test set è: {mean_ts_error}")

  final = Ensemble([model1, model2 ,model3 ,model4 ,model5 ,model6, model7])

  print(f"Ensamble MEE over training: {model_loss(final, MEE(), X, y)}")
  print(f"Ensamble MEE over test: {model_loss(final, MEE(), test_x, test_y)}")

  blind_test_predicted = final.predict(blind_test_x)

  with open("space-stars_ML-CUP21-TS.csv", "w") as f:
    header_blind_str = None

    with open("header_blind_test.txt", "r") as h:
      header_blind_str = h.readlines()

    header_blind_str += ["\n"]
    print(header_blind_str)

    f.writelines(header_blind_str)

    writer = csv.writer(f)
    for i, row in enumerate(blind_test_predicted, 1):
      row = row.tolist()
      row.insert(0,i)
      writer.writerow(row)
