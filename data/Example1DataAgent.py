import random
import numpy as np
import pandas as pd
from core.DataSet import DataSet
from data.IDataAgent import IDataAgent
from sklearn.model_selection import train_test_split

class Example1DataAgent(IDataAgent):
    '''
    Este metodo preparara los datos para poder entrenarlos.
    La columna A es la primera columna
    La columna B es la segunda columna
    La columna C es la tercera columna
    La columna Result es el resultado final.
    '''
    def retrieveDataFromSource(self):
        #Este metodo calculara un simple algoritmo con resultado que sea par o impar, algo bastante simple.
        columnA = []
        columnB = []
        columnC = []
        result = []

        for index  in range(1, 2000):
            a_object = random.randint(1, 10)
            b_object = random.randint(1, 26)
            c_object = random.randint(1, 75)
            r_object = (int(a_object + (2*b_object) -c_object)) % 2
            columnA.append(a_object)
            columnB.append(b_object)
            columnC.append(c_object)
            result.append(r_object)
        
        d = {'x':columnA, 'y':columnB, 'z':columnC, 'resultado':result}
        self.data = pd.DataFrame(data = d)

    def createTrainingDataSet(self, test_size):
        # Specify the data 
        X=self.data.ix[:,1:4]
        # Specify the target labels and flatten the array
        Y= np.ravel(self.data.resultado)
        # Split the data up in train and test sets
        self.trainDataSet = DataSet()
        self.validationDataSet = DataSet()
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        self.trainDataSet.setTrainSet(X_train)
        self.trainDataSet.setResultSet(y_train)
        self.trainDataSet.scaleData()

        self.validationDataSet.setTrainSet(X_test)
        self.validationDataSet.setResultSet(y_test)
        self.validationDataSet.scaleData()