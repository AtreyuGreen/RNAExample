import os  
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
        self.csvPath = "test1.csv"
        if(not os.path.isfile(self.csvPath)):
            print "############ CREANDO DATOS..............."
            self.data = pd.read_csv(self.csvPath) 
        else:
            print "&&&&&&&&&&&&& CARGANDO DATOS................."
            self._createCsv()


    def createTrainingDataSet(self, test_size):
        # Specify the data 
        X=self.data.ix[:,0:3]
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

    def _createCsv(self):
        #Este metodo calculara un simple algoritmo con resultado que sea par o impar, algo bastante simple.
        columnA = []
        columnB = []
        columnC = []
        result = []

        for index  in range(1, 2000):
            a_object = random.randint(1, 10)
            b_object = random.randint(1, 26)
            c_object = random.randint(1, 75)
            #Analizamos si todos los valores estan debajo del mayor medio.
            #Si es asi consideramos que la etiqueta es por debajo de la media.
            if a_object < 5 and b_object < 13 and c_object < 37:
                r_object = 1
            else:
                r_object = 0

            columnA.append(a_object)
            columnB.append(b_object)
            columnC.append(c_object)
            result.append(r_object)
        
        d = {'C1':columnA, 'C2':columnB, 'C3':columnC, 'resultado':result}
        self.data = pd.DataFrame(data = d)
        self.data.to_csv (self.csvPath, index = None, header=True) #Don't forget to add '.csv' at the end of the path        