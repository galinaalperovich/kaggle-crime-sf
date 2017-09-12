README

Semestral work on BIA subject
May 2016, Alperovich Galina,
shchegal@fel.cvut.cz
Kaggle task "Crime classification in San Francisco"
URL: https://www.kaggle.com/c/sf-crime

---------------------------
Structure of the project: 
---------------------------

* sfcrime_data_preparation.ipynb - reading, vizualization, feature extraction and saving to .h5 file

* different_classifiers.ipynb - Logistic Regression, Random Forests, XgBoost, KNN and it's comparison. Parameters for RF, XGB and KNN are selected after experementations 

* model/ folder - list of NN models which were invlolved to experementations. Rest of the models are still running.

* Ensembling: for all classifiers in different_classifiers.ipynb we calculated classes probabilities for each of k folds (allypred variable), that means it can be used as an input feature for next classifiers (stacking) or also it can be blended with another probabilities from other classifiers. Blending - composition function under several probabilities results from different classifiers. Blending and Stacking cross-validation procedures are still running. 

* Intermediate results of logloss for different NN is here (file will be updated since there are many additional models are running with different data and settings):
https://docs.google.com/spreadsheets/d/1qeAQOeDu3sI8lirf49QD7eeigWs_MNZbDhA1oXElQ94/edit?usp=sharing


