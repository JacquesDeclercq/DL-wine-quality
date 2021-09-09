# DL-wine-quality

Foobar is a Python library for dealing with word pluralization.

## The Mission
Wine tasting has been around since the creation of wine itself. However, in the modern era, more importance has been given to drinking a good wine, e.g. a French Bordeaux. France has always been hailed as the land of the wine. However, during the Judgment of Paris in 1976, a Californian wine scored better than a French wine which led to the increase in popularity of Californian wine.

## Mission objectives
Use a deep learning library.
Prepare a data set for a machine learning model.
Put together a simple neural network &
tune parameters of a neural network.

Additionaly see and compare with different classifiers.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install recommended libraries and use the documentation to create a Tensorflow environment

```bash
pip install pandas
pip install numpy
pip install tensorflow.keras
pip install seaborn
pip install sci-kit learn

```

## Usage

```python
import tensorflow as tf
from tensorflow import keras
import EarlyStopping()

# Make sure to follow the TensorFlow Keras Documentation for a valid installation)
```

## Steps
1. Feature Engineering : Convert the target to Binary (0 & 1) as the quality of wines are determined by a score from 0 to 10. Upsampling the dataset with more quality wines (copy & append). And convert the target type to Int64.

1.1 Plots & Heatmap OG Data

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/Screenshot%202021-09-09%20at%2014.35.33.png" width="450">

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/Screenshot%202021-09-09%20at%2015.32.49.png" width="450">

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/heatmap.png" width="450">

2. Base Deep Learning Model :
```python

Model: "sequential_27"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_95 (Dense)             (None, 20)                240       
_________________________________________________________________
dense_96 (Dense)             (None, 40)                840       
_________________________________________________________________
dense_97 (Dense)             (None, 10)                410       
_________________________________________________________________
dense_98 (Dense)             (None, 2)                 22        
=================================================================
Total params: 1,512
Trainable params: 1,512
Non-trainable params: 0
```

Scores : 
loss: 0.2524
accuracy: 0.5464
val_accuracy: 0.5252

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/conf_base_model.png" width="350">

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/conf2_base_model.png" width="350">


### Base Model with different loss parameter (categorical_crossentropy)

```python


Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_16 (Dense)             (None, 20)                240       
_________________________________________________________________
dense_17 (Dense)             (None, 40)                840       
_________________________________________________________________
dense_18 (Dense)             (None, 10)                410       
_________________________________________________________________
dense_19 (Dense)             (None, 1)                 11        
=================================================================
Total params: 1,501
Trainable params: 1,501
Non-trainable params: 0
_________________________________________________________________
```     
Scores : 
loss: loss: 0.00
accuracy: 0.4895
val_accuracy: 0.5148

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/conf_cross_base_model.png" width="350">

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/conf_cross_base2_model.png" width="350">
##

### Comparing Both Models' Losses
<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/compare_base_model.png" width="350">


## Model with Standardized Data
```python
Model: "sequential_33"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_118 (Dense)            (None, 20)                240       
_________________________________________________________________
batch_normalization_8 (Batch (None, 20)                80        
_________________________________________________________________
dense_119 (Dense)            (None, 40)                840       
_________________________________________________________________
dense_120 (Dense)            (None, 2)                 82        
=================================================================
Total params: 1,242
Trainable params: 1,202
Non-trainable params: 40
```
<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/models_conf.png" width="350">

Scores : 
loss: 0.2524
accuracy: 0.5464
val_accuracy: 0.5252
##

## Model with Standardized Data & Categorical Targets (best score)
``` python
Model: "sequential_36"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_127 (Dense)            (None, 20)                240       
_________________________________________________________________
batch_normalization_11 (Batc (None, 20)                80        
_________________________________________________________________
dense_128 (Dense)            (None, 40)                840       
_________________________________________________________________
dense_129 (Dense)            (None, 1)                 41        
=================================================================
Total params: 1,201
Trainable params: 1,161
Non-trainable params: 40
```
## Standardized Model Comparison :
<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/Screenshot%202021-09-09%20at%2015.47.24.png" width="350">

Scores : 
loss: 0.1012
accuracy: 0.8769
val_accuracy: 0.8265

## Different Classifiers
RandomForest : TEST ACCURACY SCORE : 0.80058 %
TRAIN ACCURACY SCORE : 0.81578 %

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/importance.png" width="350">

Decision Tree :
Tuned Decision Tree Parameters: {'criterion': 'entropy', 'max_depth': None, 'max_features': 8, 'min_samples_leaf': 6}
Best score is 0.8391742239744602

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/confdec.png" width="350">

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/confdec2.png" width="350">

Logistic Regression : ROC AUC = 0.801
AUC scores computed using 5-fold cross-validation: [0.78847069 0.84290438 0.7993698  0.74866219 0.67036768] 0.8012114702596516
AUC: 0.8012114702596516

<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/conflog.png" width="350">

KNN Neighbours : Test Accuracy Score : 0.79
Training Accuracy Score : 0.82
##
<img src="https://github.com/JacquesDeclercq/DL-wine-quality/blob/main/images/Screenshot%202021-09-09%20at%2015.48.38.png" width="350">

## Last Words

Fun project, discovering how to build, fit and the parameters of a deep learning model. As it has been said ; 'Use the tools in your belt wisely. This case was best handled by the RandomForestClassifier.

##

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
