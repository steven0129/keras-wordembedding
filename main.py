import lib.word as word
import lib.ml as ml
import numpy as np
import csv

data, labels = word.trainToken(
    './data/rawwords.csv', numOfData=1, numOfClass=2)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nbValidationSample = int(0.2 * data.shape[0])

Xtrain = data[:-nbValidationSample]
Ytrain = labels[:-nbValidationSample]
Xtest = data[:nbValidationSample]
Ytest = labels[:nbValidationSample]

# build model
myModel = ml.dlModel(inputDim=61, activation='softmax',
                   loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
myModel.fit(Xtrain, Ytrain, epochs=400, validation_split=0.2, verbose=1)

# evaluate model
loss, accuracy = myModel.evaluate(Xtest, Ytest)
print(accuracy)
