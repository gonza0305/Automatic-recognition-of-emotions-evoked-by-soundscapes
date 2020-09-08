# gonza0305.github.io

![uc3m](http://materplat.org/wp-content/uploads/LogoUC3M.jpg)

# Master thesis:

# Automatic recognition of emotions evoked by soundscapes


Performance evaluation using different machine learning models, using the Emo-soundscapes database for predicting the arousal valence values using the 1213 audio files. Additional information of the database can be found at http://metacreation.net/emo-soundscapes/.

The models that have been evaluated are:

* Support Vector Machines for Regression (SVR)
* Random Forest (RF)
* Gaussian Process Regressor (GPR)
* Extreme Gradient Boosting (XGBoost)
* Convolutional Neural Network (CNN) 

## Installing / Getting started

All the project was developed using Python and the installed libraries requiered are:

```shell
pip install librosa
pip install xgboost
pip install keras
pip install tensorflow
pip install GPy 
```
> Additional information:
The code above is included in the .ipynb file of the project, and it install the required libraries for executing the project

For the correct operation of the program, it is recommended to have downloaded and extracted the Emo-soundscapes database with the link specified above. And that it is present in the same folder in which the project is located.


## Features

* Evaluation of machine learning models using handcrafted audio features
* Analysis and selection of the most important features of sounds in order to improve the accuracy in the prediction of valence and arousal
* Design and tuning of a CNN using the log-mel spectrogram of the audio files

> Additional information:
All the evaluated models are located in the same file
