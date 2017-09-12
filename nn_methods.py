from keras.callbacks import Callback
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import time
from numpy.ma import mean
import pandas as pd

__author__ = 'galya'

import numpy as np


def get_index(Xtrain_d, k):
    indtrain1 = []
    indtrain2 = []

    for ind in range(k):
        all_idx = np.arange(len(Xtrain_d))
        np.random.shuffle(all_idx)  # we need to randomize indexes because data is ordered
        indtrain1.append(all_idx[all_idx % k != ind])
        indtrain2.append(all_idx[all_idx % k == ind])

    return indtrain1, indtrain2


# Callback for NN learning (Early Stop + save intermediate model)
class classify_on_validation_set(Callback):
    def __init__(self, filepath, validation_data=(), patience=10):
        super(Callback, self).__init__()
        self.patience = patience
        self.X_val, self.y_val = validation_data  # tuple of validation X and y
        self.best_logloss = 10.0
        self.best_acc = 10.0
        self.wait = 0  # counter for patience
        self.filepath = filepath
        self.best_rounds = 1
        self.counter = 0

    def on_epoch_end(self, epoch, logs={}):
        self.counter += 1
        current = self.model.evaluate(self.X_val, self.y_val, batch_size=self.X_val.shape[0])
        current_logloss = current[0]
        current_acc = current[1]
        print 'On test set: loss: %s, acc: %s' % (current_logloss, current_acc)
        # if improvement over best....
        if current_logloss < self.best_logloss:
            self.best_logloss = current_logloss
            self.best_acc = current_acc
            self.best_rounds = self.counter
            self.wait = 0
            self.model.save_weights(self.filepath, overwrite=True)
            print self.filepath
        else:
            print "== No improvement =="
            if self.wait >= self.patience:  # no more patience, retrieve best model
                self.model.stop_training = True
                print('Best number of rounds: %d \nBest logloss: %f \nBest acc: %s \n' % (
                    self.best_rounds, self.best_logloss, self.best_acc))
                self.model.load_weights(self.filepath)

            self.wait += 1  # incremental the number of times without improvement


def galnn(model, ind1, ind2, x_train, y_train, nb_epoch, filepath):
    # define train and test set
    data = x_train[ind1, :]
    n1 = ind1.shape[0]
    labels = to_categorical(y_train[ind1].reshape((n1, 1)), 39)

    n2 = ind2.shape[0]
    X_test = x_train[ind2, :]
    Y_test = to_categorical(y_train[ind2].reshape((n2, 1)), 39)
    val_call = classify_on_validation_set(validation_data=(X_test, Y_test), patience=5,
                                          filepath=filepath)  # instantiate object
    # fit model
    print "== Model fit =="
    # early_stop = EarlyStopping(monitor='categorical_crossentropy', patience=4, verbose=0, mode='auto')
    model.fit(data, labels, nb_epoch=nb_epoch, batch_size=500000, callbacks=[val_call])

    # Predict on test and calculate loss with arr
    print "== Model evaluate on test =="
    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=200000)

    loss = loss_and_metrics[0]
    acc = loss_and_metrics[1]
    print "== Model predict on test =="
    y_pred = model.predict(X_test)
    print "== Done =="

    return loss, acc, y_pred


def run_model(Xtrain_d, ytrain, indtrain1, indtrain2, list_of_nn, nb_epoch, model_name):
    """
    :type nb_epoch: int
    :type list_of_nn: list of Models
    """
    k = len(list_of_nn)
    t0 = time.time()
    accs = []
    losses = []
    allypred = np.zeros(
        (len(Xtrain_d), 39))  # we want to predict with NN new features = values of predected prob. by NN

    for ind in range(k):
        print "===== FOLD %s/4 ======" % ind
        model = list_of_nn[ind]
        ind_train = indtrain1[ind]  # ind1
        ind_test = indtrain2[ind]  # ind2
        loss, acc, y_pred = galnn(model, ind_train, ind_test, Xtrain_d, ytrain, nb_epoch, "%s_%s" % (model_name, ind))
        accs.append(acc)
        losses.append(loss)
        allypred[ind_test, :] = y_pred
        print "Loss: %s, Acc: %s, Time: %s\n" % (loss, acc, (time.time() - t0) / 60)

    avg_loss = mean(losses)
    avg_accs = mean(accs)
    print "Avg.loss: %s, Avg.acc: %s" % (avg_loss, avg_accs)
    return avg_loss, avg_accs, allypred


def write_result(y_t, filename, dtest, le_cat):
    result = pd.DataFrame(y_t, index=dtest.Id)
    result.columns = le_cat.classes_
    result.to_csv(filename + '.csv', float_format='%.5f')





