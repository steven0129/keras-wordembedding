from keras.models import Sequential
from keras.layers import Dense


def dlModel(inputDim, activation, loss, optimizer, metrics):
    model = Sequential()
    model.add(Dense(30, input_dim=inputDim, activation=activation))
    model.add(Dense(5, activation=activation))
    model.compile(loss=loss,
                  optimizer=optimizer, metrics=metrics)

    return model
