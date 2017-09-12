import os
import h5py
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
from nn_methods import run_model, get_index

__author__ = 'galya'
# learning rate 0.05

model_name = os.path.basename(__file__)
model_name = model_name.replace('.py', '')

k = 5
nb_epoch = 100

print("Load data from h5...")

f = h5py.File("SFData.hdf5", "r")
Xtrain_d = f["X_train"][:]
ytrain = f["y_train"][:]
Xtest_d = f["X_test"][:]
dtest = f["X_test_ID"][:]
le_cat_class = f["le_cat_classes"][:]


# Indexes for k-fold cross-validation

indtrain1, indtrain2 = get_index(Xtrain_d, k)

list_of_nn_models = []

for i in range(k):
    model = Sequential()
    model.add(Dense(250, input_dim=46, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(100, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(39, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    list_of_nn_models.append(model)



g = h5py.File("data_%s.hdf5" % model_name, "w")

print("Calculate average loss and accuracy...")

# Calculate average loss and accuracy with k-fold cross-validation with early stop learning
avg_loss, avg_accs, allypred = run_model(Xtrain_d, ytrain, indtrain1, indtrain2, list_of_nn_models, nb_epoch, model_name)

print avg_loss, avg_accs, allypred

print("Save to file...")
g.create_dataset("allypred", data=allypred)
g.create_dataset("avg_loss", data=avg_loss)
g.create_dataset("avg_accs", data=avg_accs)
g.close()