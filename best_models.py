import matplotlib.pyplot as plt

from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import read_cup
from mlprj.model_selection import grid_search_cv, split_train_params
from mlprj.feed_forward import Network, Layer

from mlprj.losses import *
from mlprj.regularizers import *
from mlprj.initializers import *
from mlprj.utility import model_loss
from mlprj.ensamble import *


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


X, test_x, y, test_y, _ = read_cup()

epochs = 500
model1 = cup_build_model_1(learning_rate= 0.15, alpha = 0.125, lambdareg = 0.00005)
history1 = model1.training((X, y), (test_x, test_y), epochs, batch_size = 32, early_stopping = 20, verbose = False)
print("Model 1 trained")
model2 = cup_build_model_1(learning_rate= 0.15, alpha = 0.225, lambdareg = 0.0001, decay = (250, 0.005))
history2 = model2.training((X, y), (test_x, test_y), epochs, batch_size = 64, early_stopping = 20, verbose = False)
print("Model 2 trained")
model3 = cup_build_model_2(learning_rate= 0.075, alpha = 0.23, lambdareg = 0.00003)
history3 = model3.training((X, y), (test_x, test_y), epochs, batch_size = 32, early_stopping = 20, verbose = False)
print("Model 3 trained")
model4 = cup_build_model_1(learning_rate= 0.4, alpha = 0.1, lambdareg = 0.000075, decay = (250, 0.01))
history4 = model4.training((X, y), (test_x, test_y), epochs, batch_size = 32, early_stopping = 20, verbose = False)
print("Model 4 trained")
model5 = cup_build_model_2(learning_rate= 0.2, alpha = 0.175, lambdareg = 0.000025)
history5 = model5.training((X, y),(test_x, test_y), epochs, batch_size = 32, early_stopping = 20, verbose = False)
print("Model 5 trained")
model6 = cup_build_model_1(learning_rate= 0.4, alpha = 0.225, lambdareg = 0.00005, decay = (250, 0.01))
history6 = model6.training((X, y), (test_x, test_y), epochs, batch_size = 64, early_stopping = 20, verbose = False)
print("Model 6 trained")


plt.plot(history1["loss_tr"], label = "training error")
plt.plot(history1["loss_vl"], linestyle = "dashed",  label = "internal test set error")
plt.legend()
plt.show()

plt.plot(history2["loss_tr"], label = "training error")
plt.plot(history2["loss_vl"], linestyle = "dashed",  label = "internal test set error")
plt.legend()
plt.show()

plt.plot(history3["loss_tr"], label = "training error")
plt.plot(history3["loss_vl"], linestyle = "dashed",  label = "internal test set error")
plt.legend()
plt.show()

plt.plot(history4["loss_tr"], label = "training error")
plt.plot(history4["loss_vl"], linestyle = "dashed",  label = "internal test set error")
plt.legend()
plt.show()

plt.plot(history5["loss_tr"], label = "training error")
plt.plot(history5["loss_vl"], linestyle = "dashed",  label = "internal test set error")
plt.legend()
plt.show()

plt.plot(history6["loss_tr"], label = "training error")
plt.plot(history6["loss_vl"], linestyle = "dashed",  label = "internal test set error")
plt.legend()
plt.show()


tr_results = [history1["loss_tr"][-1], history2["loss_tr"][-1], history3["loss_tr"][-1], history4["loss_tr"][-1], history5["loss_tr"][-1], history6["loss_tr"][-1]]
ts_results = [history1["loss_vl"][-1], history2["loss_vl"][-1], history3["loss_vl"][-1], history4["loss_vl"][-1], history5["loss_vl"][-1], history6["loss_vl"][-1]]
mean_tr_error = np.mean(np.array(tr_results))
sd_tr_error = np.std(np.array(tr_results))
mean_ts_error = np.mean(np.array(ts_results))
sd_ts_error = np.std(np.array(ts_results))

print(f"L'errore medio nel training set è: {mean_tr_error}")
print(f"La deviazione standard nel training set è: {sd_tr_error}")
print(f"L'errore medio nell' internal test set è: {mean_ts_error}")
print(f"La deviazione standard nel test set è: {sd_ts_error}")


#final = Ensemble(model1, model2 ,model3 ,model4 ,model5 ,model6)

