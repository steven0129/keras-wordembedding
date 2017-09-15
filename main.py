from sklearn.model_selection import KFold
import lib.word as word
import lib.ml as ml
import numpy as np
import csv

data, labels = word.trainToken(
    './data/rawwords.csv', numOfData=1, numOfClass=2)
cvscores = []

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
X = data[indices]
Y = labels[indices]
kf = KFold(n_splits=10, shuffle=True)

for trainIndex, testIndex in kf.split(X):
    # build model
    myModel = ml.dlModel(inputDim=data.shape[1], activation='softmax',
                         loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train model
    myModel.fit(X[trainIndex], Y[trainIndex], epochs=400,
                validation_split=0.2, verbose=1)

    # evaluate model
    scores = myModel.evaluate(X[testIndex], Y[testIndex], verbose=0)
    cvscores.append(scores[1] * 100)

    # print accuracy
    print("%s: %.f%%" % (myModel.metrics_names[1], scores[1] * 100))

print("%.2f (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
