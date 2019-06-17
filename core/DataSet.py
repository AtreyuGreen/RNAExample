from sklearn.preprocessing import StandardScaler

'''
Identifica un conjunto de datos que va a ser utilizado por la RNA.
'''
class DataSet(object):

    '''
    Establece el conjunto de entrenamiento.
    @param [DataFrame] trainSet: Es el conjunto de entrenamiento
    '''
    def setTrainSet(self, trainSet):
        self.trainSet = trainSet

    '''
    Establece el conjunto de validacion
    @param [DataFrame] testSet: Son los resultados del conjunto de entrenamiento
    '''
    def setResultSet(self, testSet):
        self.resultSet = testSet

    '''
    Escala los datos para que puedan ser utilizados correctamente en la base de datos.
    '''
    def scaleData(self):
        scaler = StandardScaler().fit(self.trainSet) # Define the scaler 
        self.trainSet = scaler.transform(self.trainSet)    # Scale the train set    
        #self.resultSet = scaler.transform(self.resultSet) # Scale the test set