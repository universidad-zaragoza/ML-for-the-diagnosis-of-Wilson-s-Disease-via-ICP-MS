# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:45:53 2022

@author: Javi
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from analysis import _predictive_entropy, _expected_entropy, bayesian_predictions, reliability_diagram
import config
import matplotlib.pyplot as plt
import os
#####################################################################
#input_coulmns is the number of columns in the input data
# 2 columns: "Cu total (ppb)" and "Cu exch"
# 3 columns: "Cu total (ppb)", "Cu exch" and "REC"
# 5 columns: "Cu total (ppb)", "Cu exch", "REC", "Delta (Cu tot)" and  "Delta (Cu Exch)"
# Cross_val_iteration is the number of crossvalidation experiment (from 1 to 4)


#############################################################
def cross_val(Cross_val_iteration, input_coulmns):
        
    if Cross_val_iteration == 1:
        file2load_train = "Datos_nor_crossval_train_1.csv"
        file2load_test = "Datos_nor_crossval_test_1.csv"
        Wilson_train_labels = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1], dtype="int32")
        Wilson_test_labels = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1], dtype="int32")
    elif Cross_val_iteration == 2:
        file2load_train = "Datos_nor_crossval_train_2.csv"
        file2load_test = "Datos_nor_crossval_test_2.csv"
        Wilson_train_labels = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1], dtype="int32")
        Wilson_test_labels = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1], dtype="int32")
    elif Cross_val_iteration == 3:
        file2load_train = "Datos_nor_crossval_train_3.csv"
        file2load_test = "Datos_nor_crossval_test_3.csv"
        Wilson_train_labels = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1], dtype="int32")
        Wilson_test_labels = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1], dtype="int32")
    elif Cross_val_iteration == 4:
        file2load_train = "Datos_nor_crossval_train_4.csv"
        file2load_test = "Datos_nor_crossval_test_4.csv"
        Wilson_train_labels = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1], dtype="int32")
        Wilson_test_labels = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1], dtype="int32")
        
    Wilson_train = pd.read_csv(
        file2load_train, 
        sep=';',
        names=["Cu total (ppb)", "Delta (Cu tot)", "Cu exch", "REC", "Sexo",
               "Delta (Cu Exch)"],
        dtype='float64')
    
    Wilson_test = pd.read_csv(
        file2load_test,
        sep=';',
        names=["Cu total (ppb)", "Delta (Cu tot)", "Cu exch", "REC", "Sexo",
               "Delta (Cu Exch)"],
        #names=["Cu total (ppb)", "Cu exch", "REC"],
        dtype='float64')
        
    if input_coulmns == 3:
        Wilson_train = Wilson_train.drop(['Delta (Cu tot)', 'Sexo', 'Delta (Cu Exch)'], axis=1)
        Wilson_test = Wilson_test.drop(['Delta (Cu tot)', 'Sexo', 'Delta (Cu Exch)'], axis=1)
    elif input_coulmns == 5:
        Wilson_train = Wilson_train.drop(['Sexo'], axis=1)
        Wilson_test = Wilson_test.drop(['Sexo'], axis=1)
    elif input_coulmns == 2:
        Wilson_train = Wilson_train.drop(['Delta (Cu tot)', 'Sexo', 'Delta (Cu Exch)', 'REC'], axis=1)
        Wilson_test = Wilson_test.drop(['Delta (Cu tot)', 'Sexo', 'Delta (Cu Exch)', 'REC'], axis=1)
            
    Wilson_features = Wilson_train.copy()
    Wilson_features = np.array(Wilson_features)
    
    Wilson_test = Wilson_test.copy()
    Wilson_test = np.array(Wilson_test)
    
    
        
    ##############MODEL Definition
    l1_n = 4
    l2_n = 4
    l3_n = 4
    l4_n = 2
    num_classes = 2
    num_features = input_coulmns
    learning_rate = 0.001
    epocas = 2000;
    num_NN_models = 50;
    models = list()
    
    for _ in range(num_NN_models):
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        inputs = keras.Input(shape=(num_features,), name="input_data")
        x = layers.Dense(l1_n, activation="relu", name="dense_1")(inputs)
        x = layers.Dense(l2_n, activation="relu", name="dense_2")(x)
        x = layers.Dense(l3_n, activation="relu", name="dense_3")(x)
            
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                             loss='categorical_crossentropy', metrics=['accuracy'])
        
        print("Fit model on training data")
        history = model.fit(
            Wilson_features,
            tf.one_hot(Wilson_train_labels, num_classes),
             epochs=epocas,
            verbose = 0,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
           #validation_data=(x_val, y_val),
        )
        
        models.append(model)
        
    # Evaluate the model on the test data using `evaluate`
    results_train = list()
    results_test = list()
    print("Evaluate on train data")
    for i in range(num_NN_models):
        results = models[i].evaluate(Wilson_features, tf.one_hot(Wilson_train_labels, num_classes))
        results_train.append(results)
        #print(i, ": train loss, train acc:", results_2col)
        results = models[i].evaluate(Wilson_test, tf.one_hot(Wilson_test_labels, num_classes))
        #print(i, ": test loss, test acc:", results_2col)
        results_test.append(results)
        
     
    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions")
    
    #num_NN_models = 10;
    test_size = 14
    NN_models = 50
    predictions_array_NN = np.zeros((NN_models, test_size, num_classes))
        
    for i in range(50):
        predictions_array_NN[i,:,:] = bayesian_predictions(models[i], Wilson_test, samples=1)
    
    pred_entropy = _predictive_entropy(predictions_array_NN)
    entropy_mean_2C = pred_entropy.mean()
    
    predictions_mean_NN = np.mean(predictions_array_NN, axis=0) # Bayesian samples average
    
    #################### Probability figure
    eje_y0 = predictions_mean_NN[:,0]
    eje_y1 = predictions_mean_NN[:,1]
    num_groups = len(eje_y0)
    index = np.arange(num_groups)
    
    plt.bar(index, eje_y1, label='Probability Wilson')
    plt.bar(index, eje_y0, label='Probability No Wilson',  bottom=eje_y1)
    plt.title('Salida NN Wilson/No Wilson 4/4/2 2000epoch LR 0.001')
    plt.legend()
     
    plt.show()
    #################### Predictive_entropy figure
            
    eje_y0 = pred_entropy
    num_groups = len(eje_y0)
    index = np.arange(num_groups)
       
    plt.bar(index, eje_y0, label='Predictive Entropy') 
    
    plt.title('Entropy ensemble 50 NN 4/4/2 2000epoch LR 0.001')
    plt.legend()
     
    plt.show()
    
    entropy_mean= pred_entropy.mean()
    
    #################################################
    # Saving the models and the results
    np.save('Ensemble_50_4_4_4_2000epoch_'+str(input_coulmns)+'columns_crossval_'+str(Cross_val_iteration)+'.npy', predictions_array_NN)
    np.savetxt('Probability_map_'+str(input_coulmns)+'columns_crossval_'+str(Cross_val_iteration)+'.csv', predictions_mean_NN, delimiter=";", fmt='%.18e')
    np.savetxt('Pred_entropy_'+str(input_coulmns)+'columns_crossval_'+str(Cross_val_iteration)+'.csv', pred_entropy, delimiter=";", fmt='%.18e')
    
    