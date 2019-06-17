from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import random

class NeuronalNetwork(object):

    def __init__(self):
        self.model = Sequential() # Creamos el modelo sequencial
        self._layers = []

    '''
    setLayer
    @description insertara las capas dentro del modelo
    @param [NeuronalNetworkLayer]: Es el conjunto de capa neuronal que se insertara en el modelo.
    '''
    def setLayer(self, layers):
        self._layers = layers

    '''
    Creara el modelo con las capas establecidas
    '''
    def create(self):
        index = 0
        for layer in self._layers:
            if(index == 0):
                self.model.add(Dense(layer.numberNode, activation=layer.activationFunction, input_shape=(layer.numberNode,))) # Add an input layer     
            else:
                self.model.add(Dense(layer.numberNode, activation=layer.activation_hidden))

    '''
    Entrena la red neuronal y devolvera el score del dataset de validacion
    @param [Model] model: Es el modelo ya entrenado.
    @param [DataSet] dataset: Es el conjunto y los resultados de entrenamiento
    @param [String] loss: Es el nombre del metodo para evaluar el error. 'binary_crossentropy', 'mean_squared_error'
    @param [String] optimizer: Es el nombre del optimizador para evaluar el error. 'sgd', 'adam'
    '''
    def train(self, dataset, loss='binary_crossentropy', optimizer='sgd'):
        self.model.compile(loss=loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])
        self.model.fit(dataset.trainSet, dataset.resultSet, epochs=20, batch_size=1, verbose=1)
    
    '''
    Evalua el modelo que ha sido entrenado con datos de validacion
    @param [DataSet] dataset: Es el conjunto y los resultados de validacion
    @result: Devuelve la precision del modelo.
    '''
    def evaluate_model(self, dataset):
        return self.model.evaluate(dataset.trainSet, dataset.resultSet, verbose=1)

    '''
    Ejecutara el modelo y devolvera la prediccion de los datos seleccionados
    @result: Devuelve la prediccion de los datos
    '''
    def predict(self, dataset):
        return model.predict(dataset.trainSet)

