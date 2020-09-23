#!/usr/bin/env python
# coding: utf-8

# ![uc3m](http://materplat.org/wp-content/uploads/LogoUC3M.jpg)
# 
# 
# 
# ### Automatic recognition of emotions evoked by soundscapes
# 
# 
# ### Gonzalo Lencina Lorenzon
# 
# 
# 
# 
# 

# #### Required installations:

# In[49]:


get_ipython().system('pip install librosa')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install GPy ')


# #### Required Python libraries 

# In[1]:


import os

#Models
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import GPy
from tensorflow import keras
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import  Conv2D, MaxPooling2D

#Metrics

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Utils
from IPython.display import display
import librosa.display
import numpy  as np
import matplotlib.pyplot as plt
import librosa  # package for speech and audio analysis
import pylab as pl
import pandas as pd
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.preprocessing import Normalizer

#Configuration
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# ### Data reading and preproccesing

# In[2]:



XData = pd.read_csv ( 'Emo-Soundscapes/Emo-Soundscapes/Emo-Soundscapes-Features/Normalized_Features.csv')
Yvalence = pd.read_csv ( 'Emo-Soundscapes/Emo-Soundscapes/Emo-Soundscapes-Ratings/Valence.csv', header = None )
Yarousal = pd.read_csv ( 'Emo-Soundscapes/Emo-Soundscapes/Emo-Soundscapes-Ratings/Arousal.csv' , header = None )


XData = XData.iloc[: , 1:XData.shape[1] ]

Audio_descriptors = Yarousal.iloc[: , 0:1 ]

Yarousal = Yarousal.iloc[: , 0:Yarousal.shape[1] ]
Yvalence = Yvalence.iloc[: , 0:Yvalence.shape[1] ]

Audio_descriptors = np.squeeze(np.asarray(Audio_descriptors))
Yarousal = np.squeeze(np.asarray(Yarousal))
Yvalence = np.squeeze(np.asarray(Yvalence))


#We eliminated features whose variance is lower than a threshold (0.02).
selector = VarianceThreshold(0.02)
selector.fit(XData)
mask = selector.get_support()

XData = XData.iloc[:,mask]

print(XData.shape)


# In[3]:


XData


# In[3]:


Kfolds_cv = ShuffleSplit(n_splits=10, test_size=.20, random_state=42)
Kfolds_cv2 = ShuffleSplit(n_splits=10, test_size=.20, random_state=42)

#Function to compute the average permutation feature importance of the partitions for each feature
def compute_feature_importance_of_folds(feature_importance_folds):
    feature_importance = []
    for i in range(np.shape(feature_importance_folds)[1]):
        add = 0
        for j in range(np.shape(feature_importance_folds)[0]):
            add = add + feature_importance_folds[j][i]
        feature_importance.append(add/(np.shape(feature_importance_folds)[0]))
    return feature_importance


# ## SVR:

# In[4]:


YvalenceSVR = pd.DataFrame(Yvalence) 
YarousalSVR = pd.DataFrame(Yarousal) 

#Define the params grids to search
param_grid_valence_SVR = [
                       {'kernel': ['rbf'], 'gamma': [ 5,3,2,1.5, 1,0.9,0.8,0.7,0.6,0.5,0.4,0.3, 1e-1], 
                        'C': [0.1,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,5]}
                     ]
param_grid_arousal_SVR = [
                       {'kernel': ['rbf'], 'gamma': [10, 1,0.5,0.4,0.3,0.2,0.1,0.05,0.01,1e-2], 
                        'C': [5,6,7,8,9,10,11,12,13,14,15]}
                     ]


# ### SVR with the 39 selected features

# In[5]:



#Arrays to store the results for each of the folds
R2_valence_SVR_AllF = []
MSE_valence_SVR_AllF = []
R2_arousal_SVR_AllF = []
MSE_arousal_SVR_AllF = []

Feature_importance_train_valence_SVR = []
Feature_importance_test_valence_SVR = []
Feature_importance_train_arousal_SVR = []
Feature_importance_test_arousal_SVR = []
iteration = 0
start = time.time()

for train_index, test_index in Kfolds_cv2.split(XData):

    print("ITERATION: " , iteration)
    
    X_train_SVR, X_test_SVR = XData.iloc[ train_index , :], XData.iloc[ test_index , : ]
    y_train_valence_SVR, y_test_valence_SVR = YvalenceSVR.iloc[train_index,:],  YvalenceSVR.iloc[test_index , :]
    y_train_arousal_SVR, y_test_arousal_SVR = YarousalSVR.iloc[train_index,:],  YarousalSVR.iloc[test_index, :]
    
    print("Train set shape:")
    print(X_train_SVR.shape)
    print("Test set shape:")
    print(X_test_SVR.shape)

    #VALENCE
    CV_SVR_valence = GridSearchCV( estimator = SVR() , param_grid=param_grid_valence_SVR, 
                                  cv= Kfolds_cv , n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_SVR_valence.fit(X_train_SVR, y_train_valence_SVR.iloc[:,1] )

    SVR_final_valence = CV_SVR_valence.best_estimator_

    SVR_y_predicted_valence = SVR_final_valence.predict(X_test_SVR)

    result_valence_test = permutation_importance(SVR_final_valence, X_test_SVR, y_test_valence_SVR.iloc[:,1], 
                                                 n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    result_valence_train = permutation_importance(SVR_final_valence, X_train_SVR, y_train_valence_SVR.iloc[:,1], 
                                                  n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    
    Feature_importance_train_valence_SVR.append(result_valence_train.importances_mean)
    Feature_importance_test_valence_SVR.append(result_valence_test.importances_mean)
    
    R2score = r2_score(y_test_valence_SVR.iloc[:,1] , SVR_y_predicted_valence)
    MSEscore = mean_squared_error( y_test_valence_SVR.iloc[:,1] , SVR_y_predicted_valence)
    
    MSE_valence_SVR_AllF.append( MSEscore )
    R2_valence_SVR_AllF.append( R2score )
    
    print('Best params: {} '.format( CV_SVR_valence.best_params_))
    print("R2 score: " , R2score)

    #AROUSAL
    CV_SVR_arousal = GridSearchCV( estimator = SVR() , param_grid=param_grid_arousal_SVR,
                                  cv= Kfolds_cv ,  n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_SVR_arousal.fit(X_train_SVR, y_train_arousal_SVR.iloc[:,1] )

    SVR_final_arousal = CV_SVR_arousal.best_estimator_

    SVR_y_predicted_arousal = SVR_final_arousal.predict(X_test_SVR)

    result_arousal_test = permutation_importance(SVR_final_arousal, X_test_SVR, y_test_arousal_SVR.iloc[:,1], 
                                                 n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    result_arousal_train = permutation_importance(SVR_final_arousal, X_train_SVR, y_train_arousal_SVR.iloc[:,1], 
                                                  n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    
    Feature_importance_train_arousal_SVR.append(result_arousal_train.importances_mean)
    Feature_importance_test_arousal_SVR.append(result_arousal_test.importances_mean)
    
    R2score = r2_score(y_test_arousal_SVR.iloc[:,1] , SVR_y_predicted_arousal)
    MSEscore = mean_squared_error(y_test_arousal_SVR.iloc[:,1] , SVR_y_predicted_arousal)
    
    MSE_arousal_SVR_AllF.append(MSEscore)
    R2_arousal_SVR_AllF.append(R2score)
    
    print('Best params: {} '.format( CV_SVR_arousal.best_params_))
    print("R2 score: " , R2score)
    iteration = iteration +1
    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[6]:


print("Valence R2 SVR: {0:.3f}".format(np.mean(R2_valence_SVR_AllF)) + " std: {0:.3f}".format(np.std(R2_valence_SVR_AllF)))
print("Arousal R2 SVR: {0:.3f}".format(np.mean(R2_arousal_SVR_AllF)) + " std: {0:.3f}".format(np.std(R2_arousal_SVR_AllF)))

print("Valence MSE SVR: {0:.3f}".format(np.mean(MSE_valence_SVR_AllF)) + " std: {0:.3f}".format(np.std(MSE_valence_SVR_AllF)))
print("Arousal MSE SVR: {0:.3f}".format(np.mean(MSE_arousal_SVR_AllF)) + " std: {0:.3f}".format(np.std(MSE_arousal_SVR_AllF)))


# In[4]:



final_feature_importance_train_valence_SVR = np.array(compute_feature_importance_of_folds(Feature_importance_train_valence_SVR))
final_feature_importance_test_valence_SVR = np.array(compute_feature_importance_of_folds(Feature_importance_test_valence_SVR))
final_feature_importance_train_arousal_SVR = np.array(compute_feature_importance_of_folds(Feature_importance_train_arousal_SVR))
final_feature_importance_test_arousal_SVR = np.array(compute_feature_importance_of_folds(Feature_importance_test_arousal_SVR))

print(final_feature_importance_train_valence_SVR.shape)
print(final_feature_importance_train_valence_SVR)


# In[17]:


importance_sorted_idx_train_valence_SVR = np.argsort(final_feature_importance_train_valence_SVR)
indices_train_valence_SVR = np.arange(0, len(final_feature_importance_train_valence_SVR)) + 0.5

importance_sorted_idx_test_valence_SVR = np.argsort(final_feature_importance_test_valence_SVR)
indices_test_valence_SVR = np.arange(0, len(final_feature_importance_test_valence_SVR)) + 0.5


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11, 6))
ax1.barh(indices_train_valence_SVR,
         final_feature_importance_train_valence_SVR[importance_sorted_idx_train_valence_SVR], height=0.7)

ax1.set_yticklabels(XData.columns[importance_sorted_idx_train_valence_SVR])
ax1.set_yticks(indices_train_valence_SVR)
ax1.set_ylim((0, len(final_feature_importance_train_valence_SVR)))
ax1.set_ylabel("Features")
ax1.set_xlabel("Score")
ax1.set_title("Valence feature importance, SVR (train set)")

ax2.barh(indices_test_valence_SVR,
         final_feature_importance_test_valence_SVR[importance_sorted_idx_test_valence_SVR], height=0.7)
ax2.set_yticklabels(XData.columns[importance_sorted_idx_test_valence_SVR])
ax2.set_yticks(indices_test_valence_SVR)
ax2.set_ylim((0, len(final_feature_importance_test_valence_SVR)))
ax2.set_ylabel("Features")
ax2.set_xlabel("Score")
ax2.set_title("Valence feature importance, SVR (test set)")
fig.tight_layout()
plt.show()

importance_sorted_idx_train_arousal_SVR = np.argsort(final_feature_importance_train_arousal_SVR)
indices_train_arousal_SVR = np.arange(0, len(final_feature_importance_train_arousal_SVR)) + 0.5

importance_sorted_idx_test_arousal_SVR = np.argsort(final_feature_importance_test_arousal_SVR)
indices_test_arousal_SVR = np.arange(0, len(final_feature_importance_test_arousal_SVR)) + 0.5

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11, 6))
ax1.barh(indices_train_arousal_SVR,
         final_feature_importance_train_arousal_SVR[importance_sorted_idx_train_arousal_SVR], height=0.7)

ax1.set_yticklabels(XData.columns[importance_sorted_idx_train_arousal_SVR])
ax1.set_yticks(indices_train_arousal_SVR)
ax1.set_ylim((0, len(final_feature_importance_train_arousal_SVR)))
ax1.set_ylabel("Features")
ax1.set_xlabel("Score")
ax1.set_title("Arousal feature importance, SVR (train set)")

ax2.barh(indices_test_arousal_SVR,
         final_feature_importance_test_arousal_SVR[importance_sorted_idx_test_arousal_SVR], height=0.7)
ax2.set_yticklabels(XData.columns[importance_sorted_idx_test_arousal_SVR])
ax2.set_yticks(indices_test_arousal_SVR)
ax2.set_ylim((0, len(final_feature_importance_test_arousal_SVR)))
ax2.set_ylabel("Features")
ax2.set_xlabel("Score")
ax2.set_title("Arousal feature importance, SVR (test set)")
fig.tight_layout()
plt.show()


# The subsets of the most important features are created based on the results of the previous graphs:

# In[18]:


XData_10_features_valence_SVR = ['loudness_std','spread_mean','flatness_mean',
                        'rolloff_mean', 'eventdensity_mean', 'inharmonicity_std',
                        'chromagram_mean_8','entropy_mean','chromagram_std_8','chromagram_mean_11'
                        ]

XData_5_features_valence_SVR = ['loudness_std','spread_mean','flatness_mean',
                        'rolloff_mean','eventdensity_mean']

XData_10_features_arousal_SVR = ['loudness_std','flatness_mean','entropy_mean',
                        'rolloff_mean', 'spread_mean', 'centroid_std', 'pitch_std',
                        'mfcc_std_2','chromagram_std_11','chromagram_mean_3']

XData_5_features_arousal_SVR = ['loudness_std','flatness_mean','entropy_mean',
                        'rolloff_mean','spread_mean']


# ### SVR with the 10 most important features

# In[19]:




#This time the same process will be executed but in the 10 most important features only

R2_valence_SVR_10F = []
MSE_valence_SVR_10F = []
R2_arousal_SVR_10F = []
MSE_arousal_SVR_10F = []

start = time.time()
iteration= 0
for train_index, test_index in Kfolds_cv2.split(XData):

    print("ITERATION: ",iteration)
    
    X_train_SVR, X_test_SVR = XData.iloc[train_index,:], XData.iloc[test_index,:]
    
    X_train_valence_SVR, X_test_valence_SVR = X_train_SVR.loc[:,XData_10_features_valence_SVR], X_test_SVR.loc[:,XData_10_features_valence_SVR]
    X_train_arousal_SVR, X_test_arousal_SVR = X_train_SVR.loc[:,XData_10_features_arousal_SVR], X_test_SVR.loc[:,XData_10_features_arousal_SVR]
    
    y_train_valence_SVR, y_test_valence_SVR = YvalenceSVR.iloc[train_index,:],  YvalenceSVR.iloc[test_index , :]
    y_train_arousal_SVR, y_test_arousal_SVR = YarousalSVR.iloc[train_index,:],  YarousalSVR.iloc[test_index, :]
    
    print("X train valence shape: " , X_train_valence_SVR.shape)
    print("X test valence shape: " , X_test_valence_SVR.shape)
    print("X train arousal shape: " , X_train_arousal_SVR.shape)
    print("X test arousal shape: " , X_test_arousal_SVR.shape)
    
    
    #VALENCE
    CV_SVR_valence = GridSearchCV( estimator = SVR() , param_grid=param_grid_valence_SVR, 
                                  cv= Kfolds_cv , n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_SVR_valence.fit(X_train_valence_SVR, y_train_valence_SVR.iloc[:,1] )

    SVR_final_valence = CV_SVR_valence.best_estimator_

    SVR_y_predicted_valence = SVR_final_valence.predict(X_test_valence_SVR)

    R2score = r2_score(y_test_valence_SVR.iloc[:,1] , SVR_y_predicted_valence)
    MSEscore = mean_squared_error( y_test_valence_SVR.iloc[:,1] , SVR_y_predicted_valence)
    
    MSE_valence_SVR_10F.append(MSEscore)
    R2_valence_SVR_10F.append(R2score)
    
    print('Best params: {} '.format( CV_SVR_valence.best_params_))
    print("R2 score: " , R2score)

    #AROUSAL
    CV_SVR_arousal = GridSearchCV( estimator = SVR() , param_grid=param_grid_arousal_SVR,
                                  cv= Kfolds_cv ,  n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_SVR_arousal.fit(X_train_arousal_SVR, y_train_arousal_SVR.iloc[:,1] )

    SVR_final_arousal = CV_SVR_arousal.best_estimator_

    SVR_y_predicted_arousal = SVR_final_arousal.predict(X_test_arousal_SVR)

    R2score = r2_score(y_test_arousal_SVR.iloc[:,1] , SVR_y_predicted_arousal)
    MSEscore = mean_squared_error(y_test_arousal_SVR.iloc[:,1] , SVR_y_predicted_arousal)
    
    MSE_arousal_SVR_10F.append(MSEscore)
    R2_arousal_SVR_10F.append(R2score)
    
    print('Best params: {} '.format( CV_SVR_arousal.best_params_))
    print("R2 score: " , R2score)
    iteration = iteration +1
    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[20]:


print("Valence R2 SVR: {0:.3f}".format(np.mean(R2_valence_SVR_10F)) + " std: {0:.3f}".format(np.std(R2_valence_SVR_10F)))
print("Arousal R2 SVR: {0:.3f}".format(np.mean(R2_arousal_SVR_10F)) + " std: {0:.3f}".format(np.std(R2_arousal_SVR_10F)))

print("Valence MSE SVR: {0:.3f}".format(np.mean(MSE_valence_SVR_10F)) + " std: {0:.3f}".format(np.std(MSE_valence_SVR_10F)))
print("Arousal MSE SVR: {0:.3f}".format(np.mean(MSE_arousal_SVR_10F)) + " std: {0:.3f}".format(np.std(MSE_arousal_SVR_10F)))


# ### SVR with the 5 most important features

# In[21]:



#This time the same process will be executed but in the 5 most important features only
R2_valence_SVR_5F = []
MSE_valence_SVR_5F = []
R2_arousal_SVR_5F = []
MSE_arousal_SVR_5F = []

start = time.time()
iteration= 0

for train_index, test_index in Kfolds_cv2.split(XData):

    print("ITERATION: ",iteration)
    
    X_train_SVR, X_test_SVR = XData.iloc[train_index,:], XData.iloc[test_index,:]
    
    X_train_valence_SVR, X_test_valence_SVR = X_train_SVR.loc[:,XData_5_features_valence_SVR], X_test_SVR.loc[:,XData_5_features_valence_SVR]
    X_train_arousal_SVR, X_test_arousal_SVR = X_train_SVR.loc[:,XData_5_features_arousal_SVR], X_test_SVR.loc[:,XData_5_features_arousal_SVR]
    
    y_train_valence_SVR, y_test_valence_SVR = YvalenceSVR.iloc[train_index,:],  YvalenceSVR.iloc[test_index , :]
    y_train_arousal_SVR, y_test_arousal_SVR = YarousalSVR.iloc[train_index,:],  YarousalSVR.iloc[test_index, :]
    
    print("X train valence shape: " , X_train_valence_SVR.shape)
    print("X test valence shape: " , X_test_valence_SVR.shape)
    print("X train arousal shape: " , X_train_arousal_SVR.shape)
    print("X test arousal shape: " , X_test_arousal_SVR.shape)
    
    
    #VALENCE
    CV_SVR_valence = GridSearchCV( estimator = SVR() , param_grid=param_grid_valence_SVR, 
                                  cv= Kfolds_cv , n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_SVR_valence.fit(X_train_valence_SVR, y_train_valence_SVR.iloc[:,1] )

    SVR_final_valence = CV_SVR_valence.best_estimator_

    SVR_y_predicted_valence = SVR_final_valence.predict(X_test_valence_SVR)

    R2score = r2_score(y_test_valence_SVR.iloc[:,1] , SVR_y_predicted_valence)
    MSEscore = mean_squared_error( y_test_valence_SVR.iloc[:,1] , SVR_y_predicted_valence)
    
    MSE_valence_SVR_5F.append(MSEscore)
    R2_valence_SVR_5F.append(R2score)
    
    print('Best params: {} '.format( CV_SVR_valence.best_params_))
    print("R2 score: " , R2score)

    #AROUSAL
    CV_SVR_arousal = GridSearchCV( estimator = SVR() , param_grid=param_grid_arousal_SVR,
                                  cv= Kfolds_cv ,  n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_SVR_arousal.fit(X_train_arousal_SVR, y_train_arousal_SVR.iloc[:,1] )

    SVR_final_arousal = CV_SVR_arousal.best_estimator_

    SVR_y_predicted_arousal = SVR_final_arousal.predict(X_test_arousal_SVR)

    R2score = r2_score(y_test_arousal_SVR.iloc[:,1] , SVR_y_predicted_arousal)
    MSEscore = mean_squared_error(y_test_arousal_SVR.iloc[:,1] , SVR_y_predicted_arousal)
    
    MSE_arousal_SVR_5F.append(MSEscore)
    R2_arousal_SVR_5F.append(R2score)
    
    print('Best params: {} '.format( CV_SVR_arousal.best_params_))
    print("R2 score: " , R2score)
    iteration = iteration +1
    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[22]:


print("Valence R2 SVR: {0:.3f}".format(np.mean(R2_valence_SVR_5F)) + " std: {0:.3f}".format(np.std(R2_valence_SVR_5F)))
print("Arousal R2 SVR: {0:.3f}".format(np.mean(R2_arousal_SVR_5F)) + " std: {0:.3f}".format(np.std(R2_arousal_SVR_5F)))

print("Valence MSE SVR: {0:.3f}".format(np.mean(MSE_valence_SVR_5F)) + " std: {0:.3f}".format(np.std(MSE_valence_SVR_5F)))
print("Arousal MSE SVR: {0:.3f}".format(np.mean(MSE_arousal_SVR_5F)) + " std: {0:.3f}".format(np.std(MSE_arousal_SVR_5F)))


# ## RF:

# In[23]:


YvalenceRF = pd.DataFrame(Yvalence) 
YarousalRF = pd.DataFrame(Yarousal) 
param_grid_valence_RF = {
    'bootstrap': [True],
    'max_depth': [ 100, 110,120],
    'max_features': [  5, 6, 7],
    'min_samples_leaf': [1,2, 3],
    'min_samples_split': [3,4,5],
    'n_estimators': [1500, 2000,300]
}
param_grid_arousal_RF = {
    'bootstrap': [True],
    'max_depth': [50, 60, 70],
    'max_features': [5, 6, 7],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [5,6,7],
    'n_estimators': [ 100,200, 300]
}


# ### RF with the 39 selected features

# In[24]:



R2_valence_RF_AllF = []
MSE_valence_RF_AllF = []
R2_arousal_RF_AllF = []
MSE_arousal_RF_AllF = []
Feature_importance_train_valence_RF = []
Feature_importance_test_valence_RF = []
Feature_importance_train_arousal_RF = []
Feature_importance_test_arousal_RF = []
start = time.time()
iteration = 0

for train_index, test_index in Kfolds_cv2.split(XData):

 
    print("ITERATION: " , iteration)
    
    X_train_RF, X_test_RF = XData.iloc[ train_index , :], XData.iloc[ test_index , : ]
    y_train_valence_RF, y_test_valence_RF = YvalenceRF.iloc[train_index,:],  YvalenceRF.iloc[test_index , :]
    y_train_arousal_RF, y_test_arousal_RF = YarousalRF.iloc[train_index,:],  YarousalRF.iloc[test_index, :]
    
    print("Train set shape:")
    print(X_train_RF.shape)
    print("Test set shape:")
    print(X_test_RF.shape)
    
    #VALENCE
    CV_RF_valence = GridSearchCV( estimator = RandomForestRegressor() , param_grid=param_grid_valence_RF, 
                                  cv= Kfolds_cv , n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_RF_valence.fit(X_train_RF, y_train_valence_RF.iloc[:,1] )

    RF_final_valence = CV_RF_valence.best_estimator_

    RF_y_predicted_valence = RF_final_valence.predict(X_test_RF)

    result_valence_test = permutation_importance(RF_final_valence, X_test_RF, y_test_valence_RF.iloc[:,1], 
                                                 n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    result_valence_train = permutation_importance(RF_final_valence, X_train_RF, y_train_valence_RF.iloc[:,1], 
                                                  n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    
    Feature_importance_train_valence_RF.append(result_valence_train.importances_mean)
    Feature_importance_test_valence_RF.append(result_valence_test.importances_mean)
    
    R2score = r2_score(y_test_valence_RF.iloc[:,1] , RF_y_predicted_valence)
    MSEscore = mean_squared_error( y_test_valence_RF.iloc[:,1] , RF_y_predicted_valence) 
    MSE_valence_RF_AllF.append(MSEscore)
    R2_valence_RF_AllF.append(R2score)
    
    print('Best params: {} '.format(CV_RF_valence.best_params_))
    print("R2 score: " , R2score)
    
    #AROUSAL
    CV_RF_arousal = GridSearchCV( estimator = RandomForestRegressor() , param_grid=param_grid_arousal_RF,
                                  cv= Kfolds_cv ,  n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_RF_arousal.fit(X_train_RF, y_train_arousal_RF.iloc[:,1] )

    RF_final_arousal = CV_RF_arousal.best_estimator_

    RF_y_predicted_arousal = RF_final_arousal.predict(X_test_RF)
    
    result_arousal_test = permutation_importance(RF_final_arousal, X_test_RF, y_test_arousal_RF.iloc[:,1], 
                                                 n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    result_arousal_train = permutation_importance(RF_final_arousal, X_train_RF, y_train_arousal_RF.iloc[:,1], 
                                                  n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    
    Feature_importance_train_arousal_RF.append(result_arousal_train.importances_mean)
    Feature_importance_test_arousal_RF.append(result_arousal_test.importances_mean)

    
    R2score = r2_score(y_test_arousal_RF.iloc[:,1] , RF_y_predicted_arousal)
    MSEscore = mean_squared_error( y_test_arousal_RF.iloc[:,1] , RF_y_predicted_arousal) 
    
    MSE_arousal_RF_AllF.append(MSEscore)
    R2_arousal_RF_AllF.append(R2score)
    print('Best params: {} '.format( CV_RF_arousal.best_params_))
    print("R2 score: " , R2score)
    iteration = iteration +1

   
    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[25]:


print("Valence R2 RF: {0:.3f}".format(np.mean(R2_valence_RF_AllF)) + " std {0:.3f}".format(np.std(R2_valence_RF_AllF)))
print("Arousal R2 RF: {0:.3f}".format(np.mean(R2_arousal_RF_AllF)) + " std {0:.3f}".format(np.std(R2_arousal_RF_AllF)))

print("Valence MSE RF: {0:.3f}".format(np.mean(MSE_valence_RF_AllF)) + " std {0:.3f}".format(np.std(MSE_valence_RF_AllF)))
print("Arousal MSE RF: {0:.3f}".format(np.mean(MSE_arousal_RF_AllF)) + " std {0:.3f}".format(np.std(MSE_arousal_RF_AllF)))


# In[26]:


final_feature_importance_train_valence_RF = np.array(compute_feature_importance_of_folds(Feature_importance_train_valence_RF))
final_feature_importance_test_valence_RF = np.array(compute_feature_importance_of_folds(Feature_importance_test_valence_RF))
final_feature_importance_train_arousal_RF = np.array(compute_feature_importance_of_folds(Feature_importance_train_arousal_RF))
final_feature_importance_test_arousal_RF = np.array(compute_feature_importance_of_folds(Feature_importance_test_arousal_RF))

print(final_feature_importance_train_valence_RF.shape)
print(final_feature_importance_train_valence_RF)


# In[27]:


importance_sorted_idx_train_valence_RF = np.argsort(final_feature_importance_train_valence_RF)
indices_train_valence_RF = np.arange(0, len(final_feature_importance_train_valence_RF)) + 0.5

importance_sorted_idx_test_valence_RF = np.argsort(final_feature_importance_test_valence_RF)
indices_test_valence_RF = np.arange(0, len(final_feature_importance_test_valence_RF)) + 0.5


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11, 6))
ax1.barh(indices_train_valence_RF,
         final_feature_importance_train_valence_RF[importance_sorted_idx_train_valence_RF], height=0.7)

ax1.set_yticklabels(XData.columns[importance_sorted_idx_train_valence_RF])
ax1.set_yticks(indices_train_valence_RF)
ax1.set_ylim((0, len(final_feature_importance_train_valence_RF)))
ax1.set_ylabel("Features")
ax1.set_xlabel("Score")
ax1.set_title("Valence feature importance, RF (train set)")

ax2.barh(indices_test_valence_RF,
         final_feature_importance_test_valence_RF[importance_sorted_idx_test_valence_RF], height=0.7)
ax2.set_yticklabels(XData.columns[importance_sorted_idx_test_valence_RF])
ax2.set_yticks(indices_test_valence_RF)
ax2.set_ylim((0, len(final_feature_importance_test_valence_RF)))
ax2.set_ylabel("Features")
ax2.set_xlabel("Score")
ax2.set_title("Valence feature importance, RF (test set)")
fig.tight_layout()
plt.show()

importance_sorted_idx_train_arousal_RF = np.argsort(final_feature_importance_train_arousal_RF)
indices_train_arousal_RF = np.arange(0, len(final_feature_importance_train_arousal_RF)) + 0.5

importance_sorted_idx_test_arousal_RF = np.argsort(final_feature_importance_test_arousal_RF)
indices_test_arousal_RF = np.arange(0, len(final_feature_importance_test_arousal_RF)) + 0.5

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11, 6))
ax1.barh(indices_train_arousal_RF,
         final_feature_importance_train_arousal_RF[importance_sorted_idx_train_arousal_RF], height=0.7)

ax1.set_yticklabels(XData.columns[importance_sorted_idx_train_arousal_RF])
ax1.set_yticks(indices_train_arousal_RF)
ax1.set_ylim((0, len(final_feature_importance_train_arousal_RF)))
ax1.set_ylabel("Features")
ax1.set_xlabel("Score")
ax1.set_title("Arousal feature importance, RF (train set)")

ax2.barh(indices_test_arousal_RF,
         final_feature_importance_test_arousal_RF[importance_sorted_idx_test_arousal_RF], height=0.7)
ax2.set_yticklabels(XData.columns[importance_sorted_idx_test_arousal_RF])
ax2.set_yticks(indices_test_arousal_RF)
ax2.set_ylim((0, len(final_feature_importance_test_arousal_RF)))
ax2.set_ylabel("Features")
ax2.set_xlabel("Score")
ax2.set_title("Arousal feature importance, RF (test set)")
fig.tight_layout()
plt.show()


# The subsets of the most important features are created based on the results of the previous graphs:

# In[28]:


XData_10_features_valence_RF = ['loudness_std','spread_mean','flatness_mean',
                        'rolloff_mean', 'inharmonicity_std', 'entropy_mean',
                        'entropy_std','chromagram_std_10','centroid_std','chromagram_std_8'
                        ]

XData_5_features_valence_RF = ['loudness_std','spread_mean','flatness_mean',
                        'rolloff_mean','inharmonicity_std']

XData_10_features_arousal_RF = ['loudness_std','spread_mean','flatness_mean',
                        'brightness_mean','centroid_std','entropy_mean',
                        'rolloff_mean', 'inharmonicity_std','pitch_std','flatness_std']

XData_5_features_arousal_RF = ['loudness_std','spread_mean','flatness_mean',
                        'brightness_mean','centroid_std']


# ### RF with the 10 most important features

# In[29]:


#This time the same process will be executed but in the 10 most important features only

R2_valence_RF_10F = []
MSE_valence_RF_10F = []
R2_arousal_RF_10F = []
MSE_arousal_RF_10F = []

start = time.time()
iteration= 0
for train_index, test_index in Kfolds_cv2.split(XData):

    print("ITERATION: ",iteration)
    
    X_train_RF, X_test_RF = XData.iloc[train_index,:], XData.iloc[test_index,:]
    
    X_train_valence_RF, X_test_valence_RF = X_train_RF.loc[:,XData_10_features_valence_RF], X_test_RF.loc[:,XData_10_features_valence_RF]
    X_train_arousal_RF, X_test_arousal_RF = X_train_RF.loc[:,XData_10_features_arousal_RF], X_test_RF.loc[:,XData_10_features_arousal_RF]
    
    y_train_valence_RF, y_test_valence_RF = YvalenceRF.iloc[train_index,:],  YvalenceRF.iloc[test_index , :]
    y_train_arousal_RF, y_test_arousal_RF = YarousalRF.iloc[train_index,:],  YarousalRF.iloc[test_index, :]
    
    print("X train valence shape: " , X_train_valence_RF.shape)
    print("X test valence shape: " , X_test_valence_RF.shape)
    print("X train arousal shape: " , X_train_arousal_RF.shape)
    print("X test arousal shape: " , X_test_arousal_RF.shape)
    
    
    #VALENCE
    CV_RF_valence = GridSearchCV( estimator = RandomForestRegressor() , param_grid=param_grid_valence_RF, 
                                  cv= Kfolds_cv , n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_RF_valence.fit(X_train_valence_RF, y_train_valence_RF.iloc[:,1] )

    RF_final_valence = CV_RF_valence.best_estimator_

    RF_y_predicted_valence = RF_final_valence.predict(X_test_valence_RF)

    R2score = r2_score(y_test_valence_RF.iloc[:,1] , RF_y_predicted_valence)
    MSEscore = mean_squared_error( y_test_valence_RF.iloc[:,1] , RF_y_predicted_valence)
    
    MSE_valence_RF_10F.append(MSEscore)
    R2_valence_RF_10F.append(R2score)
    
    print('Best params: {} '.format( CV_RF_valence.best_params_))
    print("R2 score: " , R2score)

    #AROUSAL
    CV_RF_arousal = GridSearchCV( estimator = RandomForestRegressor() , param_grid=param_grid_arousal_RF,
                                  cv= Kfolds_cv ,  n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_RF_arousal.fit(X_train_arousal_RF, y_train_arousal_RF.iloc[:,1] )

    RF_final_arousal = CV_RF_arousal.best_estimator_

    RF_y_predicted_arousal = RF_final_arousal.predict(X_test_arousal_RF)

    R2score = r2_score(y_test_arousal_RF.iloc[:,1] , RF_y_predicted_arousal)
    MSEscore = mean_squared_error(y_test_arousal_RF.iloc[:,1] , RF_y_predicted_arousal)
    
    MSE_arousal_RF_10F.append(MSEscore)
    R2_arousal_RF_10F.append(R2score)
    
    print('Best params: {} '.format( CV_RF_arousal.best_params_))
    print("R2 score: " , R2score)
    iteration = iteration +1
    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[30]:


print("Valence R2 RF: {0:.3f}".format(np.mean(R2_valence_RF_10F)) + " std {0:.3f}".format(np.std(R2_valence_RF_10F)))
print("Arousal R2 RF: {0:.3f}".format(np.mean(R2_arousal_RF_10F)) + " std {0:.3f}".format(np.std(R2_arousal_RF_10F)))

print("Valence MSE RF: {0:.3f}".format(np.mean(MSE_valence_RF_10F)) + " std {0:.3f}".format(np.std(MSE_valence_RF_10F)))
print("Arousal MSE RF: {0:.3f}".format(np.mean(MSE_arousal_RF_10F)) + " std {0:.3f}".format(np.std(MSE_arousal_RF_10F)))


# ### RF with the 5 most important features

# In[31]:


#This time the same process will be executed but in the 5 most important features only

#Because the number of features used now is 5, the max_features values of the param grid is changed accordingly
param_grid_valence_RF_5F = {
    'bootstrap': [True],
    'max_depth': [ 100, 110,120],
    'max_features': [  3, 4, 5],
    'min_samples_leaf': [1,2, 3],
    'min_samples_split': [3,4,5],
    'n_estimators': [1500, 2000,300]
}
param_grid_arousal_RF_5F = {
    'bootstrap': [True],
    'max_depth': [50, 60, 70],
    'max_features': [3, 4, 5],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [5,6,7],
    'n_estimators': [ 100,200, 300]
}

R2_valence_RF_5F = []
MSE_valence_RF_5F = []
R2_arousal_RF_5F = []
MSE_arousal_RF_5F = []

start = time.time()
iteration= 0
for train_index, test_index in Kfolds_cv2.split(XData):

    print("ITERATION: ",iteration)
    
    X_train_RF, X_test_RF = XData.iloc[train_index,:], XData.iloc[test_index,:]
    
    X_train_valence_RF, X_test_valence_RF = X_train_RF.loc[:,XData_5_features_valence_RF], X_test_RF.loc[:,XData_5_features_valence_RF]
    X_train_arousal_RF, X_test_arousal_RF = X_train_RF.loc[:,XData_5_features_arousal_RF], X_test_RF.loc[:,XData_5_features_arousal_RF]
    
    y_train_valence_RF, y_test_valence_RF = YvalenceRF.iloc[train_index,:],  YvalenceRF.iloc[test_index , :]
    y_train_arousal_RF, y_test_arousal_RF = YarousalRF.iloc[train_index,:],  YarousalRF.iloc[test_index, :]
    
    print("X train valence shape: " , X_train_valence_RF.shape)
    print("X test valence shape: " , X_test_valence_RF.shape)
    print("X train arousal shape: " , X_train_arousal_RF.shape)
    print("X test arousal shape: " , X_test_arousal_RF.shape)
    
    
    #VALENCE
    CV_RF_valence = GridSearchCV( estimator = RandomForestRegressor() , param_grid=param_grid_valence_RF_5F, 
                                  cv= Kfolds_cv , n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_RF_valence.fit(X_train_valence_RF, y_train_valence_RF.iloc[:,1] )

    RF_final_valence = CV_RF_valence.best_estimator_

    RF_y_predicted_valence = RF_final_valence.predict(X_test_valence_RF)

    R2score = r2_score(y_test_valence_RF.iloc[:,1] , RF_y_predicted_valence)
    MSEscore = mean_squared_error( y_test_valence_RF.iloc[:,1] , RF_y_predicted_valence)
    
    MSE_valence_RF_5F.append(MSEscore)
    R2_valence_RF_5F.append(R2score)
    
    print('Best params: {} '.format( CV_RF_valence.best_params_))
    print("R2 score: " , R2score)

    #AROUSAL
    CV_RF_arousal = GridSearchCV( estimator = RandomForestRegressor() , param_grid=param_grid_arousal_RF_5F,
                                  cv= Kfolds_cv ,  n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_RF_arousal.fit(X_train_arousal_RF, y_train_arousal_RF.iloc[:,1] )

    RF_final_arousal = CV_RF_arousal.best_estimator_

    RF_y_predicted_arousal = RF_final_arousal.predict(X_test_arousal_RF)

    R2score = r2_score(y_test_arousal_RF.iloc[:,1] , RF_y_predicted_arousal)
    MSEscore = mean_squared_error(y_test_arousal_RF.iloc[:,1] , RF_y_predicted_arousal)
    
    MSE_arousal_RF_5F.append(MSEscore)
    R2_arousal_RF_5F.append(R2score)
    
    print('Best params: {} '.format( CV_RF_arousal.best_params_))
    print("R2 score: " , R2score)
    iteration = iteration +1
    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[32]:


print("Valence R2 RF: {0:.3f}".format(np.mean(R2_valence_RF_5F)) + " std {0:.3f}".format(np.std(R2_valence_RF_5F)))
print("Arousal R2 RF: {0:.3f}".format(np.mean(R2_arousal_RF_5F)) + " std {0:.3f}".format(np.std(R2_arousal_RF_5F)))

print("Valence MSE RF: {0:.3f}".format(np.mean(MSE_valence_RF_5F)) + " std {0:.3f}".format(np.std(MSE_valence_RF_5F)))
print("Arousal MSE RF: {0:.3f}".format(np.mean(MSE_arousal_RF_5F)) + " std {0:.3f}".format(np.std(MSE_arousal_RF_5F)))


# ## GPR:
# 
# 
# 

# In[23]:


YvalenceGP = pd.DataFrame(Yvalence) 
YarousalGP = pd.DataFrame(Yarousal) 


# ### GPR with the 39 selected features, comparing different kernels

# In[9]:



R2_scores_RBF_valence_AllF = []
R2_scores_LIN_valence_AllF = []
R2_scores_RQ_valence_AllF = []
R2_scores_MAT3_valence_AllF = []
R2_scores_LIN_RQ_valence_AllF = []
R2_scores_MAT3_RQ_valence_AllF = []
R2_scores_RBF_RQ_valence_AllF = []
MSE_scores_RBF_valence_AllF = []
MSE_scores_LIN_valence_AllF = []
MSE_scores_RQ_valence_AllF = []
MSE_scores_MAT3_valence_AllF = []
MSE_scores_LIN_RQ_valence_AllF = []
MSE_scores_MAT3_RQ_valence_AllF = []
MSE_scores_RBF_RQ_valence_AllF = []

R2_scores_RBF_arousal_AllF = []
R2_scores_LIN_arousal_AllF = []
R2_scores_RQ_arousal_AllF = []
R2_scores_MAT3_arousal_AllF = []
R2_scores_LIN_RQ_arousal_AllF = []
R2_scores_MAT3_RQ_arousal_AllF = []
R2_scores_RBF_RQ_arousal_AllF = []
MSE_scores_RBF_arousal_AllF = []
MSE_scores_LIN_arousal_AllF = []
MSE_scores_RQ_arousal_AllF = []
MSE_scores_MAT3_arousal_AllF = []
MSE_scores_LIN_RQ_arousal_AllF = []
MSE_scores_MAT3_RQ_arousal_AllF = []
MSE_scores_RBF_RQ_arousal_AllF = []


start = time.time()
KFolds_shuffleSplit =  ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
iteration = 0 

for train_index, test_index in KFolds_shuffleSplit.split(XData):

   print("ITERATION: ",iteration)

   X_train_GP, X_test_GP = XData.iloc[ train_index , :], XData.iloc[ test_index , : ]
   y_train_valence_GP, y_test_valence_GP = YvalenceGP.iloc[train_index,:],  YvalenceGP.iloc[test_index , :]
   y_train_arousal_GP, y_test_arousal_GP = YarousalGP.iloc[train_index,:],  YarousalGP.iloc[test_index, :]


   kernel_valence_RBF = GPy.kern.RBF( input_dim=39 , ARD = True )
   kernel_valence_LIN = GPy.kern.Linear( input_dim=39 , ARD = True )
   kernel_valence_RQ = GPy.kern.RatQuad( input_dim=39 , ARD = True )
   kernel_valence_MAT3 = GPy.kern.Matern32( input_dim=39 , ARD = True )
   kernel_valence_LIN_RQ = GPy.kern.Linear( input_dim=39 , ARD = True ) + GPy.kern.RatQuad( input_dim=39 , ARD = True )
   kernel_Valence_MAT3_RQ = GPy.kern.Matern32( input_dim=39 , ARD = True ) + GPy.kern.RatQuad( input_dim=39 , ARD = True )

   GP_model_valence_RBF = GPy.models.GPRegression(X_train_GP , y_train_valence_GP.iloc[:,1].values.reshape([-1,1]),kernel_valence_RBF)
   GP_model_valence_LIN = GPy.models.GPRegression(X_train_GP , y_train_valence_GP.iloc[:,1].values.reshape([-1,1]),kernel_valence_LIN)
   GP_model_valence_RQ = GPy.models.GPRegression(X_train_GP , y_train_valence_GP.iloc[:,1].values.reshape([-1,1]),kernel_valence_RQ)
   GP_model_valence_MAT3 = GPy.models.GPRegression(X_train_GP , y_train_valence_GP.iloc[:,1].values.reshape([-1,1]),kernel_valence_MAT3)
   GP_model_valence_LIN_RQ = GPy.models.GPRegression(X_train_GP , y_train_valence_GP.iloc[:,1].values.reshape([-1,1]),kernel_valence_LIN_RQ)
   GP_model_valence_MAT3_RQ = GPy.models.GPRegression(X_train_GP , y_train_valence_GP.iloc[:,1].values.reshape([-1,1]),kernel_Valence_MAT3_RQ)
   
   GP_model_valence_RBF.optimize(messages=False, max_iters=2000)
   GP_model_valence_LIN.optimize(messages=False, max_iters=2000)
   GP_model_valence_RQ.optimize(messages=False, max_iters=2000)
   GP_model_valence_MAT3.optimize(messages=False, max_iters=2000)
   GP_model_valence_LIN_RQ.optimize(messages=False, max_iters=2000)
   GP_model_valence_MAT3_RQ.optimize(messages=False, max_iters=2000)
   
   
   kernel_arousal_RBF = GPy.kern.RBF( input_dim=39 , ARD = True )
   kernel_arousal_LIN = GPy.kern.Linear( input_dim=39 , ARD = True )
   kernel_arousal_RQ = GPy.kern.RatQuad( input_dim=39 , ARD = True )
   kernel_arousal_MAT3 = GPy.kern.Matern32( input_dim=39 , ARD = True )
   kernel_arousal_LIN_RQ = GPy.kern.Linear( input_dim=39 , ARD = True ) + GPy.kern.RatQuad( input_dim=39 , ARD = True )
   kernel_arousal_MAT3_RQ = GPy.kern.Matern32( input_dim=39 , ARD = True ) + GPy.kern.RatQuad( input_dim=39 , ARD = True )


   GP_model_arousal_RBF = GPy.models.GPRegression(X_train_GP , y_train_arousal_GP.iloc[:,1].values.reshape([-1,1]),kernel_arousal_RBF)
   GP_model_arousal_LIN = GPy.models.GPRegression(X_train_GP , y_train_arousal_GP.iloc[:,1].values.reshape([-1,1]),kernel_arousal_LIN)
   GP_model_arousal_RQ = GPy.models.GPRegression(X_train_GP , y_train_arousal_GP.iloc[:,1].values.reshape([-1,1]),kernel_arousal_RQ)
   GP_model_arousal_MAT3 = GPy.models.GPRegression(X_train_GP, y_train_arousal_GP.iloc[:,1].values.reshape([-1,1]),kernel_arousal_MAT3)
   GP_model_arousal_LIN_RQ = GPy.models.GPRegression(X_train_GP, y_train_arousal_GP.iloc[:,1].values.reshape([-1,1]),kernel_arousal_LIN_RQ)
   GP_model_arousal_MAT3_RQ = GPy.models.GPRegression(X_train_GP, y_train_arousal_GP.iloc[:,1].values.reshape([-1,1]),kernel_arousal_MAT3_RQ)

   GP_model_arousal_RBF.optimize(messages=False , max_iters=2000)
   GP_model_arousal_LIN.optimize(messages=False, max_iters=2000)
   GP_model_arousal_RQ.optimize(messages=False, max_iters=2000)
   GP_model_arousal_MAT3.optimize(messages=False, max_iters=2000)
   GP_model_arousal_LIN_RQ.optimize(messages=False, max_iters=2000)
   GP_model_arousal_MAT3_RQ.optimize(messages=False, max_iters=2000)

   meanYtest_valence,_ = GP_model_valence_RBF.predict(X_test_GP.to_numpy())
   R2_scores_RBF_valence_AllF.append(r2_score(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))
   MSE_scores_RBF_valence_AllF.append(mean_squared_error(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))
   
   meanYtest_valence,_ = GP_model_valence_LIN.predict(X_test_GP.to_numpy())
   R2_scores_LIN_valence_AllF.append(r2_score(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))
   MSE_scores_LIN_valence_AllF.append(mean_squared_error(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))

   meanYtest_valence,_ = GP_model_valence_RQ.predict(X_test_GP.to_numpy())
   R2_scores_RQ_valence_AllF.append(r2_score(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))
   MSE_scores_RQ_valence_AllF.append(mean_squared_error(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))

   meanYtest_valence,_ = GP_model_valence_MAT3.predict(X_test_GP.to_numpy())
   R2_scores_MAT3_valence_AllF.append(r2_score(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))
   MSE_scores_MAT3_valence_AllF.append(mean_squared_error(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))

   meanYtest_valence,_ = GP_model_valence_LIN_RQ.predict(X_test_GP.to_numpy())
   R2_scores_LIN_RQ_valence_AllF.append(r2_score(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))
   MSE_scores_LIN_RQ_valence_AllF.append(mean_squared_error(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))

   meanYtest_valence,_ = GP_model_valence_MAT3_RQ.predict(X_test_GP.to_numpy())
   R2_scores_MAT3_RQ_valence_AllF.append(r2_score(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))
   MSE_scores_MAT3_RQ_valence_AllF.append(mean_squared_error(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))

   meanYtest_arousal,_ = GP_model_arousal_RBF.predict(X_test_GP.to_numpy())
   R2_scores_RBF_arousal_AllF.append(r2_score(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))
   MSE_scores_RBF_arousal_AllF.append(mean_squared_error(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))

   meanYtest_arousal,_ = GP_model_arousal_LIN.predict(X_test_GP.to_numpy())
   R2_scores_LIN_arousal_AllF.append(r2_score(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))
   MSE_scores_LIN_arousal_AllF.append(mean_squared_error(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))

   meanYtest_arousal,_ = GP_model_arousal_RQ.predict(X_test_GP.to_numpy())
   R2_scores_RQ_arousal_AllF.append(r2_score(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))
   MSE_scores_RQ_arousal_AllF.append(mean_squared_error(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))

   meanYtest_arousal,_ = GP_model_arousal_MAT3.predict(X_test_GP.to_numpy())
   R2_scores_MAT3_arousal_AllF.append(r2_score(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))
   MSE_scores_MAT3_arousal_AllF.append(mean_squared_error(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))

   meanYtest_arousal,_ = GP_model_arousal_LIN_RQ.predict(X_test_GP.to_numpy())
   R2_scores_LIN_RQ_arousal_AllF.append(r2_score(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))
   MSE_scores_LIN_RQ_arousal_AllF.append(mean_squared_error(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))

   meanYtest_arousal,_ = GP_model_arousal_MAT3_RQ.predict(X_test_GP.to_numpy())
   R2_scores_MAT3_RQ_arousal_AllF.append(r2_score(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))
   MSE_scores_MAT3_RQ_arousal_AllF.append(mean_squared_error(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))

   iteration = iteration +1

print("Time (minutes):")
print((time.time()-start)/60)
print("\n")



# In[10]:



print("Valence RBF R2: {0:.3f}".format(np.mean(R2_scores_RBF_valence_AllF)) + " std {0:.3f}".format(np.std(R2_scores_RBF_valence_AllF)))
print("Valence RBF MSE: {0:.3f}".format(np.mean(MSE_scores_RBF_valence_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_RBF_valence_AllF)))

print("Valence LIN R2: {0:.3f}".format(np.mean(R2_scores_LIN_valence_AllF)) + " std {0:.3f}".format(np.std(R2_scores_LIN_valence_AllF)))
print("Valence LIN MSE: {0:.3f}".format(np.mean(MSE_scores_LIN_valence_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_LIN_valence_AllF)))

print("Valence RQ R2: {0:.3f}".format(np.mean(R2_scores_RQ_valence_AllF)) + " std {0:.3f}".format(np.std(R2_scores_RQ_valence_AllF)))
print("Valence RQ MSE: {0:.3f}".format(np.mean(MSE_scores_RQ_valence_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_RQ_valence_AllF)))

print("Valence MAT3 R2: {0:.3f}".format(np.mean(R2_scores_MAT3_valence_AllF)) + " std {0:.3f}".format(np.std(R2_scores_MAT3_valence_AllF)))
print("Valence MAT3 MSE: {0:.3f}".format(np.mean(MSE_scores_MAT3_valence_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_MAT3_valence_AllF)))

print("Valence LIN + RQ R2: {0:.3f}".format(np.mean(R2_scores_LIN_RQ_valence_AllF)) + " std {0:.3f}".format(np.std(R2_scores_LIN_RQ_valence_AllF)))
print("Valence LIN + RQ MSE: {0:.3f}".format(np.mean(MSE_scores_LIN_RQ_valence_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_LIN_RQ_valence_AllF)))

print("Valence MAT3 + RQ R2: {0:.3f}".format(np.mean(R2_scores_MAT3_RQ_valence_AllF)) + " std {0:.3f}".format(np.std(R2_scores_MAT3_RQ_valence_AllF)))
print("Valence MAT3 + RQ MSE: {0:.3f}".format(np.mean(MSE_scores_MAT3_RQ_valence_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_MAT3_RQ_valence_AllF)))


print("Arousal RBF R2: {0:.3f}".format(np.mean(R2_scores_RBF_arousal_AllF)) + " std {0:.3f}".format(np.std(R2_scores_RBF_arousal_AllF)))
print("Arousal RBF MSE: {0:.3f}".format(np.mean(MSE_scores_RBF_arousal_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_RBF_arousal_AllF)))

print("Arousal LIN R2 : {0:.3f}".format(np.mean(R2_scores_LIN_arousal_AllF)) + " std {0:.3f}".format(np.std(R2_scores_LIN_arousal_AllF)))
print("Arousal LIN MSE: {0:.3f}".format(np.mean(MSE_scores_LIN_arousal_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_LIN_arousal_AllF)))

print("Arousal RQ R2: {0:.3f}".format(np.mean(R2_scores_RQ_arousal_AllF)) + " std {0:.3f}".format(np.std(R2_scores_RQ_arousal_AllF)))
print("Arousal RQ MSE: {0:.3f}".format(np.mean(MSE_scores_RQ_arousal_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_RQ_arousal_AllF)))

print("Arousal MAT3 R2: {0:.3f}".format(np.mean(R2_scores_MAT3_arousal_AllF)) + " std {0:.3f}".format(np.std(R2_scores_MAT3_arousal_AllF)))
print("Arousal MAT3 MSE: {0:.3f}".format(np.mean(MSE_scores_MAT3_arousal_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_MAT3_arousal_AllF)))

print("Arousal LIN + RQ R2: {0:.3f}".format(np.mean(R2_scores_LIN_RQ_arousal_AllF)) + " std {0:.3f}".format(np.std(R2_scores_LIN_RQ_arousal_AllF)))
print("Arousal LIN + RQ MSE: {0:.3f}".format(np.mean(MSE_scores_LIN_RQ_arousal_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_LIN_RQ_arousal_AllF)))

print("Arousal MAT3 + RQ R2: {0:.3f}".format(np.mean(R2_scores_MAT3_RQ_arousal_AllF)) + " std {0:.3f}".format(np.std(R2_scores_MAT3_RQ_arousal_AllF)))
print("Arousal MAT3 + RQ MSE: {0:.3f}".format(np.mean(MSE_scores_MAT3_RQ_arousal_AllF)) + " std {0:.3f}".format(np.std(MSE_scores_MAT3_RQ_arousal_AllF)))




# Compute the average lengthscale values of the RQ kernel (best kernel) for all the partitions in the cross-validation:

# In[6]:


lengthscale_values_valence = []
lengthscale_values_arousal = []

R2_valence_GP_AllF = [] 
MSE_valence_GP_AllF = []

R2_arousal_GP_AllF = []
MSE_arousal_GP_AllF = []

YvalenceGP = pd.DataFrame(Yvalence) 
YarousalGP = pd.DataFrame(Yarousal) 

start = time.time()

KFolds_shuffleSplit =  ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
iteration = 0 

for train_index, test_index in KFolds_shuffleSplit.split(XData):

    print("ITERATION: ",iteration)

    X_train_GP, X_test_GP = XData.iloc[ train_index , :], XData.iloc[ test_index , : ]
    y_train_valence_GP, y_test_valence_GP = YvalenceGP.iloc[train_index,:],  YvalenceGP.iloc[test_index, :]
    y_train_arousal_GP, y_test_arousal_GP = YarousalGP.iloc[train_index,:],  YarousalGP.iloc[test_index, :]

    print("X train  shape: " , X_train_GP.shape)
    print("X test  shape: " , X_test_GP.shape)
   
    kernel_valence_RQ = GPy.kern.RatQuad( input_dim=39 , ARD = True )
    GP_model_valence_RQ = GPy.models.GPRegression(X_train_GP , y_train_valence_GP.iloc[:,1].values.reshape([-1,1]), kernel_valence_RQ)
    GP_model_valence_RQ.optimize(messages=False, max_iters=2000)
    
    meanYtest_valence,_ = GP_model_valence_RQ.predict(X_test_GP.to_numpy())
    R2_valence_GP_AllF.append(r2_score(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))
    MSE_valence_GP_AllF.append(mean_squared_error(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))

    lengthscale_values_valence.append(GP_model_valence_RQ.RatQuad.lengthscale.values)
    
    kernel_arousal_RQ = GPy.kern.RatQuad( input_dim=39 , ARD = True )
    GP_model_arousal_RQ = GPy.models.GPRegression(X_train_GP , y_train_arousal_GP.iloc[:,1].values.reshape([-1,1]),kernel_arousal_RQ)
    GP_model_arousal_RQ.optimize(messages=False, max_iters=2000)
    
    meanYtest_arousal,_ = GP_model_arousal_RQ.predict(X_test_GP.to_numpy())
    R2_arousal_GP_AllF.append(r2_score(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))
    MSE_arousal_GP_AllF.append(mean_squared_error(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))

    lengthscale_values_arousal.append(GP_model_arousal_RQ.RatQuad.lengthscale.values)

    iteration = iteration +1

print("Time (minutes):")
print((time.time()-start)/60)
print("\n")

lengthscale_values_valence = np.array(compute_feature_importance_of_folds(np.array(lengthscale_values_valence)))
lengthscale_values_arousal = np.array(compute_feature_importance_of_folds(np.array(lengthscale_values_arousal)))


# In[7]:


print("Valence R2 RF: {0:.3f}".format(np.mean(R2_valence_GP_AllF)) + " std {0:.3f}".format(np.std(R2_valence_GP_AllF)))
print("Arousal R2 RF: {0:.3f}".format(np.mean(R2_arousal_GP_AllF)) + " std {0:.3f}".format(np.std(R2_arousal_GP_AllF)))

print("Valence MSE RF: {0:.3f}".format(np.mean(MSE_valence_GP_AllF)) + " std {0:.3f}".format(np.std(MSE_valence_GP_AllF)))
print("Arousal MSE RF: {0:.3f}".format(np.mean(MSE_arousal_GP_AllF)) + " std {0:.3f}".format(np.std(MSE_arousal_GP_AllF)))


# In[54]:



importance_sorted_idx_valence_GP = np.argsort(lengthscale_values_valence)[::-1]
indices_valence_GP = np.arange(0, len(lengthscale_values_valence)) + 0.5

importance_sorted_idx_arousal_GP = np.argsort(lengthscale_values_arousal)[::-1]
indices_arousal_GP = np.arange(0, len(lengthscale_values_arousal)) + 0.5


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11, 6))
ax1.barh(indices_valence_GP,
         lengthscale_values_valence[importance_sorted_idx_valence_GP], height=0.7)

ax1.set_yticklabels(XData.columns[importance_sorted_idx_valence_GP])
ax1.set_yticks(indices_valence_GP)
ax1.set_ylim((0, len(lengthscale_values_valence)))
ax1.set_ylabel("Features")
ax1.set_xlabel("Lengthscale")
ax1.set_title("Valence feature importance GP")

ax2.barh(indices_arousal_GP,
         lengthscale_values_arousal[importance_sorted_idx_arousal_GP], height=0.7)
ax2.set_yticklabels(XData.columns[importance_sorted_idx_arousal_GP])
ax2.set_yticks(indices_arousal_GP)
ax2.set_ylim((0, len(lengthscale_values_arousal)))
ax2.set_ylabel("Features")
ax2.set_xlabel("Lengthscale")
ax2.set_title("Arousal feature importance GP")
fig.tight_layout()
plt.show()


# The subsets of the most important features are created based on the results of the previous graphs:

# In[36]:


XData_10_features_valence_GP = ['loudness_std','entropy_mean','mfcc_std_2',
                                 'inharmonicity_std','chromagram_std_8','spread_mean',
                                 'eventdensity_mean','chromagram_center_std_5','flatness_std','chromagram_std_10'
                                ]

XData_5_features_valence_GP = ['loudness_std','entropy_mean','mfcc_std_2',
                                'inharmonicity_std','chromagram_std_8']

XData_10_features_arousal_GP = ['loudness_std','entropy_mean','flatness_mean',
                                'pitch_std','chromagram_mean_11','inharmonicity_std',
                                'mfcc_std_2','chromagram_mean_9','centroid_std','eventdensity_mean']

XData_5_features_arousal_GP = ['loudness_std','entropy_mean','flatness_mean',
                                'pitch_std','chromagram_mean_11']


# ### GPR with the 10 most importante features

# In[37]:


R2_valence_GP_10F = []
MSE_valence_GP_10F = []
R2_arousal_GP_10F = []
MSE_arousal_GP_10F = []


start = time.time()
iteration= 0
for train_index, test_index in Kfolds_cv2.split(XData):

    print("ITERATION: ",iteration)
    
    X_train_GP, X_test_GP = XData.iloc[train_index,:], XData.iloc[test_index,:]
    
    X_train_valence_GP, X_test_valence_GP = X_train_GP.loc[:,XData_10_features_valence_GP], X_test_GP.loc[:,XData_10_features_valence_GP]
    X_train_arousal_GP, X_test_arousal_GP = X_train_GP.loc[:,XData_10_features_arousal_GP], X_test_GP.loc[:,XData_10_features_arousal_GP]
    
    y_train_valence_GP, y_test_valence_GP = YvalenceGP.iloc[train_index,:],  YvalenceGP.iloc[test_index , :]
    y_train_arousal_GP, y_test_arousal_GP = YarousalGP.iloc[train_index,:],  YarousalGP.iloc[test_index, :]
    
    print("X train valence shape: " , X_train_valence_GP.shape)
    print("X test valence shape: " , X_test_valence_GP.shape)
    print("X train arousal shape: " , X_train_arousal_GP.shape)
    print("X test arousal shape: " , X_test_arousal_GP.shape)
    
    kernel_valence_RQ = GPy.kern.RatQuad( input_dim=10 , ARD = True)
    GP_model_valence_RQ = GPy.models.GPRegression(X_train_valence_GP , y_train_valence_GP.iloc[:,1].values.reshape([-1,1]),kernel_valence_RQ)
    GP_model_valence_RQ.optimize(messages=False, max_iters=2000)
   
    meanYtest_valence,_ = GP_model_valence_RQ.predict(X_test_valence_GP.to_numpy())
    R2_valence_GP_10F.append(r2_score(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))
    MSE_valence_GP_10F.append(mean_squared_error(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))

    
    kernel_arousal_RQ = GPy.kern.RatQuad(input_dim=10 , ARD = True)
    GP_model_arousal_RQ = GPy.models.GPRegression(X_train_arousal_GP , y_train_arousal_GP.iloc[:,1].values.reshape([-1,1]),kernel_arousal_RQ)
    GP_model_arousal_RQ.optimize(messages=False, max_iters=2000)
    
    
    meanYtest_arousal,_ = GP_model_arousal_RQ.predict(X_test_arousal_GP.to_numpy())
    R2_arousal_GP_10F.append(r2_score(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))
    MSE_arousal_GP_10F.append(mean_squared_error(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))


    iteration = iteration +1
    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[38]:


print("Valence R2 GPR: {0:.3f}".format(np.mean(R2_valence_GP_10F)) + " std {0:.3f}".format(np.std(R2_valence_GP_10F)))
print("Arousal R2 GPR: {0:.3f}".format(np.mean(R2_arousal_GP_10F)) + " std {0:.3f}".format(np.std(R2_arousal_GP_10F)))

print("Valence MSE GPR: {0:.3f}".format(np.mean(MSE_valence_GP_10F)) + " std {0:.3f}".format(np.std(MSE_valence_GP_10F)))
print("Arousal MSE GPR: {0:.3f}".format(np.mean(MSE_arousal_GP_10F)) + " std {0:.3f}".format(np.std(MSE_arousal_GP_10F)))


# ### GPR with the 5 most important features

# In[39]:


R2_valence_GP_5F = []
MSE_valence_GP_5F = []
R2_arousal_GP_5F = []
MSE_arousal_GP_5F = []


start = time.time()
iteration= 0
for train_index, test_index in Kfolds_cv2.split(XData):

    print("ITERATION: ",iteration)
    
    X_train_GP, X_test_GP = XData.iloc[train_index,:], XData.iloc[test_index,:]
    
    X_train_valence_GP, X_test_valence_GP = X_train_GP.loc[:,XData_5_features_valence_GP], X_test_GP.loc[:,XData_5_features_valence_GP]
    X_train_arousal_GP, X_test_arousal_GP = X_train_GP.loc[:,XData_5_features_arousal_GP], X_test_GP.loc[:,XData_5_features_arousal_GP]
    
    y_train_valence_GP, y_test_valence_GP = YvalenceGP.iloc[train_index,:],  YvalenceGP.iloc[test_index , :]
    y_train_arousal_GP, y_test_arousal_GP = YarousalGP.iloc[train_index,:],  YarousalGP.iloc[test_index, :]
    
    print("X train valence shape: " , X_train_valence_GP.shape)
    print("X test valence shape: " , X_test_valence_GP.shape)
    print("X train arousal shape: " , X_train_arousal_GP.shape)
    print("X test arousal shape: " , X_test_arousal_GP.shape)
    
    kernel_valence_RQ = GPy.kern.RatQuad( input_dim=5 , ARD = True)
    GP_model_valence_RQ = GPy.models.GPRegression(X_train_valence_GP , y_train_valence_GP.iloc[:,1].values.reshape([-1,1]),kernel_valence_RQ)
    GP_model_valence_RQ.optimize(messages=False, max_iters=2000)
   
    meanYtest_valence,_ = GP_model_valence_RQ.predict(X_test_valence_GP.to_numpy())
    R2_valence_GP_5F.append(r2_score(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))
    MSE_valence_GP_5F.append(mean_squared_error(y_test_valence_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_valence))

    
    kernel_arousal_RQ = GPy.kern.RatQuad(input_dim=5 , ARD = True)
    GP_model_arousal_RQ = GPy.models.GPRegression(X_train_arousal_GP , y_train_arousal_GP.iloc[:,1].values.reshape([-1,1]),kernel_arousal_RQ)
    GP_model_arousal_RQ.optimize(messages=False, max_iters=2000)
    
    
    meanYtest_arousal,_ = GP_model_arousal_RQ.predict(X_test_arousal_GP.to_numpy())
    R2_arousal_GP_5F.append(r2_score(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))
    MSE_arousal_GP_5F.append(mean_squared_error(y_test_arousal_GP.iloc[:,1].values.reshape([-1,1]), meanYtest_arousal))


    iteration = iteration +1
    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[40]:


print("Valence R2 GPR: {0:.3f}".format(np.mean(R2_valence_GP_5F)) + " std {0:.3f}".format(np.std(R2_valence_GP_5F)))
print("Arousal R2 GPR: {0:.3f}".format(np.mean(R2_arousal_GP_5F)) + " std {0:.3f}".format(np.std(R2_arousal_GP_5F)))

print("Valence MSE GPR: {0:.3f}".format(np.mean(MSE_valence_GP_5F)) + " std {0:.3f}".format(np.std(MSE_valence_GP_5F)))
print("Arousal MSE GPR: {0:.3f}".format(np.mean(MSE_arousal_GP_5F)) + " std {0:.3f}".format(np.std(MSE_arousal_GP_5F)))


# ### XGBOOST:

# In[57]:


YvalenceXGB = pd.DataFrame(Yvalence) 
YarousalXGB = pd.DataFrame(Yarousal) 


param_grid_valence_XGB = {
    'learning_rate': [0.1],
    'n_estimators': [ 60,67,80],
    'max_depth': [ 6,7,8],
    'min_child_weight': [ 55,63,70 ],
    'gamma': [0.3,0.43619, 0.5],
    'subsample': [0.7,0.8, 0.9],
    'colsample_bytree': [0.7, 0.795, 0.85],
    'reg_alpha': [ 1e-13, 1e-14 ,1e-15 ]
}


param_grid_arousal_XGB = {
    'learning_rate': [0.1],
    'n_estimators': [85, 92,100],
    'max_depth': [  3,4,5],
    'min_child_weight': [3,4,5],
    'gamma': [0.0,0.1,0.2],
    'subsample': [0.7,0.755 ,0.85],
    'colsample_bytree': [0.8,0.898, 0.95],
    'reg_alpha': [0.01,0.001,0.0001]
}


# ### XGBoost with the 39 selected features

# In[34]:




R2_valence_XGB_AllF = []
MSE_valence_XGB_AllF = []
R2_arousal_XGB_AllF = []
MSE_arousal_XGB_AllF = []

Feature_importance_train_valence_XGB = []
Feature_importance_test_valence_XGB = []
Feature_importance_train_arousal_XGB = []
Feature_importance_test_arousal_XGB = []
iteration = 0
start = time.time()

for train_index, test_index in Kfolds_cv2.split(XData):

 
    print("ITERATION: ",iteration)

    X_train_XGB, X_test_XGB = XData.iloc[ train_index , :], XData.iloc[ test_index , : ]
    y_train_valence_XGB, y_test_valence_XGB = YvalenceXGB.iloc[train_index,:],  YvalenceXGB.iloc[test_index,:]
    y_train_arousal_XGB, y_test_arousal_XGB = YarousalXGB.iloc[train_index,:],  YarousalXGB.iloc[test_index,:]
    
    #VALENCE
    CV_XGB_valence = GridSearchCV( estimator = XGBRegressor(objective= 'reg:squarederror', nthread=-1, 
                                  scale_pos_weight=1,seed=42) , param_grid=param_grid_valence_XGB, 
                                  cv= Kfolds_cv , n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_XGB_valence.fit(X_train_XGB, y_train_valence_XGB.iloc[:,1] )

    XGB_final_valence = CV_XGB_valence.best_estimator_

    XGB_y_predicted_valence = XGB_final_valence.predict(X_test_XGB)

    result_valence_test = permutation_importance(XGB_final_valence, X_test_XGB, y_test_valence_XGB.iloc[:,1], 
                                                 n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    result_valence_train = permutation_importance(XGB_final_valence, X_train_XGB, y_train_valence_XGB.iloc[:,1], 
                                                  n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    
    Feature_importance_train_valence_XGB.append(result_valence_train.importances_mean)
    Feature_importance_test_valence_XGB.append(result_valence_test.importances_mean)
    
       
    R2score = r2_score(y_test_valence_XGB.iloc[:,1] , XGB_y_predicted_valence) 
    MSEscore = mean_squared_error( y_test_valence_XGB.iloc[:,1] , XGB_y_predicted_valence)
    MSE_valence_XGB_AllF.append(MSEscore)
    R2_valence_XGB_AllF.append(R2score)
    
    print('Best params: {} '.format(CV_XGB_valence.best_params_))
    print("R2 score: " , R2score)
    
    #AROUSAL
    CV_XGB_arousal = GridSearchCV( estimator = XGBRegressor(objective= 'reg:squarederror', nthread=-1, 
                                  scale_pos_weight=1,seed=42) , param_grid=param_grid_arousal_XGB,
                                  cv= Kfolds_cv ,  n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_XGB_arousal.fit(X_train_XGB, y_train_arousal_XGB.iloc[:,1] )

    XGB_final_arousal = CV_XGB_arousal.best_estimator_

    XGB_y_predicted_arousal = XGB_final_arousal.predict(X_test_XGB)

    result_arousal_test = permutation_importance(XGB_final_arousal, X_test_XGB, y_test_arousal_XGB.iloc[:,1], 
                                                 n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    result_arousal_train = permutation_importance(XGB_final_arousal, X_train_XGB, y_train_arousal_XGB.iloc[:,1], 
                                                  n_repeats=40, random_state=42, n_jobs=-1, scoring = "r2")
    
    Feature_importance_train_arousal_XGB.append(result_arousal_train.importances_mean)
    Feature_importance_test_arousal_XGB.append(result_arousal_test.importances_mean)
    
    R2score = r2_score(y_test_arousal_XGB.iloc[:,1] , XGB_y_predicted_arousal)
    MSEscore = mean_squared_error(y_test_arousal_XGB.iloc[:,1] , XGB_y_predicted_arousal)
    
    MSE_arousal_XGB_AllF.append(MSEscore)
    R2_arousal_XGB_AllF.append(R2score)
    
    print('Best params: {} '.format( CV_XGB_arousal.best_params_))
    print("R2 score: " , R2score)
    iteration = iteration +1

    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[35]:


print("Valence R2 XGB: {0:.3f}".format(np.mean(R2_valence_XGB_AllF)) + " std {0:.3f}".format(np.std(R2_valence_XGB_AllF)))
print("Arousal R2 XGB: {0:.3f}".format(np.mean(R2_arousal_XGB_AllF)) + " std {0:.3f}".format(np.std(R2_arousal_XGB_AllF)))

print("Valence MSE XGB: {0:.3f}".format(np.mean(MSE_valence_XGB_AllF)) + " std {0:.3f}".format(np.std(MSE_valence_XGB_AllF)))
print("Arousal MSE XGB: {0:.3f}".format(np.mean(MSE_arousal_XGB_AllF)) + " std {0:.3f}".format(np.std(MSE_arousal_XGB_AllF)))


# In[36]:



final_feature_importance_train_valence_XGB = np.array(compute_feature_importance_of_folds(Feature_importance_train_valence_XGB))
final_feature_importance_test_valence_XGB = np.array(compute_feature_importance_of_folds(Feature_importance_test_valence_XGB))
final_feature_importance_train_arousal_XGB = np.array(compute_feature_importance_of_folds(Feature_importance_train_arousal_XGB))
final_feature_importance_test_arousal_XGB = np.array(compute_feature_importance_of_folds(Feature_importance_test_arousal_XGB))

print(final_feature_importance_train_valence_XGB.shape)
print(final_feature_importance_train_valence_XGB)


# In[37]:


importance_sorted_idx_train_valence_XGB = np.argsort(final_feature_importance_train_valence_XGB)
indices_train_valence_XGB = np.arange(0, len(final_feature_importance_train_valence_XGB)) + 0.5

importance_sorted_idx_test_valence_XGB = np.argsort(final_feature_importance_test_valence_XGB)
indices_test_valence_XGB = np.arange(0, len(final_feature_importance_test_valence_XGB)) + 0.5


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11, 6))
ax1.barh(indices_train_valence_XGB,
         final_feature_importance_train_valence_XGB[importance_sorted_idx_train_valence_XGB], height=0.7)

ax1.set_yticklabels(XData.columns[importance_sorted_idx_train_valence_XGB])
ax1.set_yticks(indices_train_valence_XGB)
ax1.set_ylim((0, len(final_feature_importance_train_valence_XGB)))
ax1.set_ylabel("Features")
ax1.set_xlabel("Score")
ax1.set_title("Valence feature importance, XGBoost (train set)")

ax2.barh(indices_test_valence_XGB,
         final_feature_importance_test_valence_XGB[importance_sorted_idx_test_valence_XGB], height=0.7)
ax2.set_yticklabels(XData.columns[importance_sorted_idx_test_valence_XGB])
ax2.set_yticks(indices_test_valence_XGB)
ax2.set_ylim((0, len(final_feature_importance_test_valence_XGB)))
ax2.set_ylabel("Features")
ax2.set_xlabel("Score")
ax2.set_title("Valence feature importance, XGBoost (test set)")
fig.tight_layout()
plt.show()

importance_sorted_idx_train_arousal_XGB = np.argsort(final_feature_importance_train_arousal_XGB)
indices_train_arousal_XGB = np.arange(0, len(final_feature_importance_train_arousal_XGB)) + 0.5

importance_sorted_idx_test_arousal_XGB = np.argsort(final_feature_importance_test_arousal_XGB)
indices_test_arousal_XGB = np.arange(0, len(final_feature_importance_test_arousal_XGB)) + 0.5

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11, 6))
ax1.barh(indices_train_arousal_XGB,
         final_feature_importance_train_arousal_XGB[importance_sorted_idx_train_arousal_XGB], height=0.7)

ax1.set_yticklabels(XData.columns[importance_sorted_idx_train_arousal_XGB])
ax1.set_yticks(indices_train_arousal_XGB)
ax1.set_ylim((0, len(final_feature_importance_train_arousal_XGB)))
ax1.set_ylabel("Features")
ax1.set_xlabel("Score")
ax1.set_title("Arousal feature importance, XGBoost (train set)")

ax2.barh(indices_test_arousal_XGB,
         final_feature_importance_test_arousal_XGB[importance_sorted_idx_test_arousal_XGB], height=0.7)
ax2.set_yticklabels(XData.columns[importance_sorted_idx_test_arousal_XGB])
ax2.set_yticks(indices_test_arousal_XGB)
ax2.set_ylim((0, len(final_feature_importance_test_arousal_XGB)))
ax2.set_ylabel("Features")
ax2.set_xlabel("Score")
ax2.set_title("Arousal feature importance, XGBoost (test set)")
fig.tight_layout()
plt.show()


# The subsets of the most important features are created based on the results of the previous graphs:

# In[55]:


XData_10_features_valence_XGB = ['loudness_std','spread_mean','inharmonicity_std',
                                 'entropy_mean','rolloff_mean','chromagram_std_10',
                                 'chromagram_std_8','entropy_std','mfcc_std_2','chromagram_mean_8']

XData_5_features_valence_XGB = ['loudness_std','spread_mean','inharmonicity_std',
                                'entropy_mean','rolloff_mean']

XData_10_features_arousal_XGB = ['loudness_std','brightness_mean','entropy_mean',
                                'spread_mean','pitch_std','inharmonicity_std',
                                'centroid_std','rolloff_mean','entropy_std' ,'chromagram_mean_7' ]

XData_5_features_arousal_XGB = ['loudness_std','brightness_mean','entropy_mean',
                                'spread_mean','pitch_std']


# ### XGBoost with the 10 most important features

# In[39]:




R2_valence_XGB_10F = []
MSE_valence_XGB_10F = []
R2_arousal_XGB_10F = []
MSE_arousal_XGB_10F = []
start = time.time()
iteration = 0

for train_index, test_index in Kfolds_cv2.split(XData):

 
    print("ITERATION: ", iteration)
    
    X_train_XGB, X_test_XGB = XData.iloc[train_index,:], XData.iloc[test_index,:]
    
    X_train_valence_XGB, X_test_valence_XGB = X_train_XGB.loc[:,XData_10_features_valence_XGB], X_test_XGB.loc[:,XData_10_features_valence_XGB]
    X_train_arousal_XGB, X_test_arousal_XGB = X_train_XGB.loc[:,XData_10_features_arousal_XGB], X_test_XGB.loc[:,XData_10_features_arousal_XGB]
    
    y_train_valence_XGB, y_test_valence_XGB = YvalenceXGB.iloc[train_index,:],  YvalenceXGB.iloc[test_index , :]
    y_train_arousal_XGB, y_test_arousal_XGB = YarousalXGB.iloc[train_index,:],  YarousalXGB.iloc[test_index, :]
    
    print("X train valence shape: " , X_train_valence_XGB.shape)
    print("X test valence shape: " , X_test_valence_XGB.shape)
    print("X train arousal shape: " , X_train_arousal_XGB.shape)
    print("X test arousal shape: " , X_test_arousal_XGB.shape)
    
    #VALENCE
    CV_XGB_valence = GridSearchCV( estimator = XGBRegressor(objective= 'reg:squarederror', nthread=-1, 
                                  scale_pos_weight=1,seed=42) , param_grid=param_grid_valence_XGB, 
                                  cv= Kfolds_cv , n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_XGB_valence.fit(X_train_XGB, y_train_valence_XGB.iloc[:,1] )

    XGB_final_valence = CV_XGB_valence.best_estimator_

    XGB_y_predicted_valence = XGB_final_valence.predict(X_test_XGB)

    
    R2score = r2_score(y_test_valence_XGB.iloc[:,1] , XGB_y_predicted_valence) 
    MSEscore = mean_squared_error( y_test_valence_XGB.iloc[:,1] , XGB_y_predicted_valence)
    MSE_valence_XGB_10F.append(MSEscore)
    R2_valence_XGB_10F.append(R2score)
    
    print('Best params: {} '.format(CV_XGB_valence.best_params_))
    print("R2 score: " , R2score)
    
    #AROUSAL
    CV_XGB_arousal = GridSearchCV( estimator = XGBRegressor(objective= 'reg:squarederror', nthread=-1, 
                                  scale_pos_weight=1,seed=42) , param_grid=param_grid_arousal_XGB,
                                  cv= Kfolds_cv ,  n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_XGB_arousal.fit(X_train_XGB, y_train_arousal_XGB.iloc[:,1] )

    XGB_final_arousal = CV_XGB_arousal.best_estimator_

    XGB_y_predicted_arousal = XGB_final_arousal.predict(X_test_XGB)

    
    R2score = r2_score(y_test_arousal_XGB.iloc[:,1] , XGB_y_predicted_arousal)
    MSEscore = mean_squared_error(y_test_arousal_XGB.iloc[:,1] , XGB_y_predicted_arousal)
    MSE_arousal_XGB_10F.append(MSEscore)
    R2_arousal_XGB_10F.append(R2score)
    print('Best params: {} '.format( CV_XGB_arousal.best_params_))
    print("R2 score: " , R2score)
    iteration = iteration +1

    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[40]:


print("Valence R2 XGB: {0:.3f}".format(np.mean(R2_valence_XGB_10F)) + " std {0:.3f}".format(np.std(R2_valence_XGB_10F)))
print("Arousal R2 XGB: {0:.3f}".format(np.mean(R2_arousal_XGB_10F)) + " std {0:.3f}".format(np.std(R2_arousal_XGB_10F)))

print("Valence MSE XGB: {0:.3f}".format(np.mean(MSE_valence_XGB_10F)) + " std {0:.3f}".format(np.std(MSE_valence_XGB_10F)))
print("Arousal MSE XGB: {0:.3f}".format(np.mean(MSE_arousal_XGB_10F)) + " std {0:.3f}".format(np.std(MSE_arousal_XGB_10F)))


# ### XGBoost with the 5 most important features

# In[58]:




R2_valence_XGB_5F = []
MSE_valence_XGB_5F = []
R2_arousal_XGB_5F = []
MSE_arousal_XGB_5F = []
start = time.time()
iteration = 0

for train_index, test_index in Kfolds_cv2.split(XData):

    print("ITERATION: " , iteration)
    
    X_train_XGB, X_test_XGB = XData.iloc[train_index,:], XData.iloc[test_index,:]
    
    X_train_valence_XGB, X_test_valence_XGB = X_train_XGB.loc[:,XData_5_features_valence_XGB], X_test_XGB.loc[:,XData_5_features_valence_XGB]
    X_train_arousal_XGB, X_test_arousal_XGB = X_train_XGB.loc[:,XData_5_features_arousal_XGB], X_test_XGB.loc[:,XData_5_features_arousal_XGB]
    
    y_train_valence_XGB, y_test_valence_XGB = YvalenceXGB.iloc[train_index,:],  YvalenceXGB.iloc[test_index , :]
    y_train_arousal_XGB, y_test_arousal_XGB = YarousalXGB.iloc[train_index,:],  YarousalXGB.iloc[test_index, :]
    
    print("X train valence shape: " , X_train_valence_XGB.shape)
    print("X test valence shape: " , X_test_valence_XGB.shape)
    print("X train arousal shape: " , X_train_arousal_XGB.shape)
    print("X test arousal shape: " , X_test_arousal_XGB.shape)
    
    #VALENCE
    CV_XGB_valence = GridSearchCV( estimator = XGBRegressor(objective= 'reg:squarederror', nthread=-1, 
                                  scale_pos_weight=1,seed=42) , param_grid=param_grid_valence_XGB, 
                                  cv= Kfolds_cv , n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_XGB_valence.fit(X_train_XGB, y_train_valence_XGB.iloc[:,1] )

    XGB_final_valence = CV_XGB_valence.best_estimator_

    XGB_y_predicted_valence = XGB_final_valence.predict(X_test_XGB)

    
    R2score = r2_score(y_test_valence_XGB.iloc[:,1] , XGB_y_predicted_valence) 
    MSEscore = mean_squared_error( y_test_valence_XGB.iloc[:,1] , XGB_y_predicted_valence)
    MSE_valence_XGB_5F.append(MSEscore)
    R2_valence_XGB_5F.append(R2score)
    
    print('Best params: {} '.format(CV_XGB_valence.best_params_))
    print("R2 score: " , R2score)
    
    #AROUSAL
    CV_XGB_arousal = GridSearchCV( estimator = XGBRegressor(objective= 'reg:squarederror', nthread=-1, 
                                  scale_pos_weight=1,seed=42) , param_grid=param_grid_arousal_XGB,
                                  cv= Kfolds_cv ,  n_jobs=-1, scoring = 'r2' , refit = 'r2' , verbose=True )

    CV_XGB_arousal.fit(X_train_XGB, y_train_arousal_XGB.iloc[:,1] )

    XGB_final_arousal = CV_XGB_arousal.best_estimator_

    XGB_y_predicted_arousal = XGB_final_arousal.predict(X_test_XGB)

    
    R2score = r2_score(y_test_arousal_XGB.iloc[:,1] , XGB_y_predicted_arousal)
    MSEscore = mean_squared_error(y_test_arousal_XGB.iloc[:,1] , XGB_y_predicted_arousal)
    MSE_arousal_XGB_5F.append(MSEscore)
    R2_arousal_XGB_5F.append(R2score)
    print('Best params: {} '.format( CV_XGB_arousal.best_params_))
    print("R2 score: " , R2score)
    iteration = iteration +1

    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[59]:


print("Valence R2 XGB: {0:.3f}".format(np.mean(R2_valence_XGB_5F)) + " std {0:.3f}".format(np.std(R2_valence_XGB_5F)))
print("Arousal R2 XGB: {0:.3f}".format(np.mean(R2_arousal_XGB_5F)) + " std {0:.3f}".format(np.std(R2_arousal_XGB_5F)))

print("Valence MSE XGB: {0:.3f}".format(np.mean(MSE_valence_XGB_5F)) + " std {0:.3f}".format(np.std(MSE_valence_XGB_5F)))
print("Arousal MSE XGB: {0:.3f}".format(np.mean(MSE_arousal_XGB_5F)) + " std {0:.3f}".format(np.std(MSE_arousal_XGB_5F)))


# ### CNN

# In[4]:


#Function to compute the average of the patches outputs
def average_output_of_patches(predicted_values):
    predicted_values = np.array(predicted_values)
    average_values = []
    i = 0
    iterations = int(predicted_values.shape[0]/6)
    for j in range( 0, iterations):
        average_aux = np.average(predicted_values[i:i+6])
        average_values.append(average_aux)
        i = i + 6
    return average_values



#Function to find the full path of a file
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

#Function to compute the log-mel patches of the spectrograms
def getNormMelPatches(y_dataset):    
    
    wst = 25e-3;          # Window size (seconds), to apply the Fourier transformation in a 25 ms window
    fpt = 10e-3;          # Frame period (seconds)
    nbands = 64;          # Number of filters in the filterbank
    
    # Read the audio file
    i = 0
    normalized_pathes = []
    y_patches = []
    
    Audio_descriptors = y_dataset[:,0]
    y_values = y_dataset[:,1]
    row = 0
    
    for audio_name in Audio_descriptors:
    
        
        full_path = find(audio_name, "./Emo-Soundscapes")

        # Read the audio file
        x, fs = librosa.load(full_path, sr=None)

        # Compute mel-spectrogram from the raw signal
        nfft = round(wst*fs); # Window size (samples)
        fp = round(fpt*fs);   # Frame period (samples)
        melfb = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=nfft, hop_length=fp, n_mels=nbands)

        # Compute log-mel spectrogram
        logmelfb = librosa.core.power_to_db(melfb, top_db=None)

        patch1 = logmelfb[:,0:100]
        transformer1 =  Normalizer().fit(patch1)
        patch1 = transformer1.transform(patch1)
        normalized_pathes.append(patch1)
        y_patches.append(y_values[row])
        
        patch2 = logmelfb[:,100:200]
        transformer2 =  Normalizer().fit(patch2)
        patch2 = transformer2.transform(patch2)
        normalized_pathes.append(patch2)
        y_patches.append(y_values[row])
        
        patch3 = logmelfb[:,200:300]
        transformer3 =  Normalizer().fit(patch3)
        patch3 = transformer3.transform(patch3)
        normalized_pathes.append(patch3)
        y_patches.append(y_values[row])
        
        patch4 = logmelfb[:,300:400]
        transformer4 =  Normalizer().fit(patch4)
        patch4 = transformer4.transform(patch4)
        normalized_pathes.append(patch4)
        y_patches.append(y_values[row])

        patch5 = logmelfb[:,400:500]
        transformer5 =  Normalizer().fit(patch5)
        patch5 = transformer5.transform(patch5)
        normalized_pathes.append(patch5)
        y_patches.append(y_values[row])
        
        patch6 = logmelfb[:,500:600]
        transformer6 =  Normalizer().fit(patch6)
        patch6 = transformer6.transform(patch6)
        normalized_pathes.append(patch6)  
        y_patches.append(y_values[row])
        
        row = row + 1
        if i ==0:

            print("Audio: " + audio_name)
            print("Window size: ")
            print(nfft)
            print("Step size: ")
            print(fp)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(logmelfb,sr=fs, hop_length=fp, y_axis='mel', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-Mel spectrogram')
            plt.show()
            i = i+1
            
    return np.array(normalized_pathes), np.array(y_patches)
        






# In[5]:


#CNN architecture
def build_cnn(activation = 'softsign',
              dropout_rate = 0.5,
              optimizer = 'Adam',
              init_mode='glorot_uniform',
              kernel_size = 3,
              param1 = 4,
              param2 = 2,
              filters1 = 64,
              neurons = 32,
              filters2 = 16
             ):
    
    model = Sequential()
    model.add(Conv2D(filters = filters1, kernel_size = (kernel_size, kernel_size), activation='relu', input_shape = (64,100,1)))
    model.add(MaxPooling2D((param1,param2), strides=(param1,param2)))

    model.add(Conv2D(filters = filters2, kernel_size = (kernel_size, kernel_size), activation='relu'  ))

    model.add(MaxPooling2D((param1,param2), strides=(param1,param2)))
    
    model.add(Conv2D( filters = filters2 , kernel_size = (kernel_size, kernel_size),  activation='relu'))
    
    model.add(Flatten())
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    #Output layer
    model.add(Dense(1, activation = 'softsign' , kernel_initializer=init_mode))
    
    model.compile(optimizer='Adam', loss='mse')
    
    return model


# In[8]:


YvalenceCNN = pd.DataFrame(Yvalence) 
YarousalCNN = pd.DataFrame(Yarousal) 


R2_valence_CNN = []
MSE_valence_CNN = []
R2_arousal_CNN = []
MSE_arousal_CNN = []
Feature_importance_train_valence_CNN = []
Feature_importance_test_valence_CNN = []
Feature_importance_train_arousal_CNN  = []
Feature_importance_test_arousal_CNN  = []

start = time.time()
iteration = 0

for train_index, test_index in Kfolds_cv2.split(XData):

    print("ITERATION: ", iteration)
    
    X_train_CNN, X_test_CNN = XData.iloc[ train_index , :], XData.iloc[ test_index , : ]
    y_train_valence_CNN, y_test_valence_CNN = YvalenceCNN.iloc[train_index,:],  YvalenceCNN.iloc[test_index , :]
    y_train_arousal_CNN, y_test_arousal_CNN = YarousalCNN.iloc[train_index,:],  YarousalCNN.iloc[test_index, :]
    
    y_train_valence_CNN = y_train_valence_CNN.to_numpy()
    y_test_valence_CNN = y_test_valence_CNN.to_numpy()
    y_train_arousal_CNN = y_train_arousal_CNN.to_numpy()
    y_test_arousal_CNN = y_test_arousal_CNN.to_numpy()

    normalized_pathes_train_valence, y_patches_train_valence = getNormMelPatches(y_train_valence_CNN)
    normalized_pathes_test_valence, y_patches_test_valence = getNormMelPatches(y_test_valence_CNN)
    normalized_pathes_train_arousal, y_patches_train_arousal = getNormMelPatches(y_train_arousal_CNN)
    normalized_pathes_test_arousal, y_patches_test_arousal = getNormMelPatches(y_test_arousal_CNN)
    
    normalized_pathes_train_valence = normalized_pathes_train_valence.reshape((5820, 64, 100, 1))
    normalized_pathes_test_valence = normalized_pathes_test_valence.reshape((1458, 64, 100, 1))
    normalized_pathes_train_arousal = normalized_pathes_train_arousal.reshape((5820, 64, 100, 1))
    normalized_pathes_test_arousal = normalized_pathes_test_arousal.reshape((1458, 64, 100, 1))
    
    #VALENCE
    model_CNN_valence = KerasRegressor(build_fn = build_cnn, verbose=True, epochs=53, batch_size=32)

    history_valence = model_CNN_valence.fit(normalized_pathes_train_valence, y_patches_train_valence)
    
    CNN_y_predicted_valence = model_CNN_valence.predict(normalized_pathes_test_valence)

    CNN_y_predicted_valence = np.array(average_output_of_patches(CNN_y_predicted_valence))
        
    MSE_valence_CNN.append( mean_squared_error( y_test_valence_CNN[:,1], CNN_y_predicted_valence) )
    R2_valence_CNN.append( r2_score(y_test_valence_CNN[:,1] , CNN_y_predicted_valence) )
    
    #AROUSAL
    model_CNN_arousal = KerasRegressor(build_fn = build_cnn, verbose=True, epochs=53, batch_size=32)

    history_arousal = model_CNN_arousal.fit(normalized_pathes_train_arousal, y_patches_train_arousal)

    CNN_y_predicted_arousal = model_CNN_arousal.predict(normalized_pathes_test_arousal)

    CNN_y_predicted_arousal = np.array(average_output_of_patches(CNN_y_predicted_arousal))
    
    MSE_arousal_CNN.append(mean_squared_error(y_test_arousal_CNN[:,1] , CNN_y_predicted_arousal))
    R2_arousal_CNN.append(r2_score(y_test_arousal_CNN[:,1] , CNN_y_predicted_arousal))
    iteration = iteration +1

    
print("Time (minutes):")
print((time.time()-start)/60)
print("\n")


# In[9]:


print("Valence R2 CNN: {0:.3f}".format(np.mean(R2_valence_CNN)) + " std {0:.3f}".format(np.std(R2_valence_CNN)))
print("Arousal R2 CNN: {0:.3f}".format(np.mean(R2_arousal_CNN)) + " std {0:.3f}".format(np.std(R2_arousal_CNN)))


print("Valence MSE CNN: {0:.3f}".format(np.mean(MSE_valence_CNN)) + " std {0:.3f}".format(np.std(MSE_valence_CNN)))
print("Arousal MSE CNN: {0:.3f}".format(np.mean(MSE_arousal_CNN)) + " std {0:.3f}".format(np.std(MSE_arousal_CNN)))


# In[ ]:


model = KerasRegressor(build_fn = build_cnn, verbose=True, epochs=53, batch_size=32)

param_grid = {
               #'kernel_size': [1,3,5],
               #'filters1': [16,32,64],
               #'filters2': [16,32,64],
               'param1': [1,2,3,4],
               'param2': [1,2,3,4]
              #'epochs':     [1000],
              #'neurons' : [32,64,128],
              #'batch_size':  [10, 20, 40, 60, 80, 100],
              #'weight_constraint' : [1,2,3,4,5],
              #'optimizer':   ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
              #'dropout_rate':[0.1,0.3,0.5],
              #'activation':  ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
              #'init_mode':   ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
              #'learning_rate' : [0.001, 0.01, 0.1, 0.2, 0.3],
              #'momentum' : [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
             }

gridsearchModel = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        scoring='r2',
        cv=5,
        refit='r2',
        verbose=True,
        n_jobs=4
        )


gridsearchModel.fit(normalized_pathes_train_arousal, y_patches_train_arousal)

print("Best: %f using %s" % (gridsearchModel.best_score_, gridsearchModel.best_params_))


# In[ ]:




