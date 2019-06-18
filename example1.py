import datetime
from core.NeuronalNetwork import NeuronalNetwork
from data.Example1DataAgent import Example1DataAgent
from core.NeuronalNetworkLayer import NeuronalNetworkLayer


def run(loss, optimizer):
    start_time = datetime.datetime.now()
    data = Example1DataAgent()
    data.retrieveDataFromSource()
    data.createTrainingDataSet(0.33)

    model = NeuronalNetwork()
    layers = [
        NeuronalNetworkLayer(3, 'relu'),
        NeuronalNetworkLayer(8, 'relu'),
        NeuronalNetworkLayer(1, 'sigmoid')
    ]

    model.setLayer(layers)
    model.create()
    model.train(data.trainDataSet, loss, optimizer)
    print(model.evaluate_model(data.validationDataSet))
    end_time = datetime.datetime.now()
    print "Tiempo de entrenamiento: "+str((end_time - start_time).seconds)+" segundos"
    #print(model.predict(data.validationDataSet))

def test_1_run():
    run('binary_crossentropy', 'sgd')

def test_2_run():
    run('mean_squared_error', 'sgd')

def test_3_run():
    run('binary_crossentropy', 'adam')

def test_4_run():
    run('mean_squared_error', 'adam')

#El primer test, se realizara con 8 neuronas hidden, con una funcion de error de 
#binary_crossentroypy y un optimizador de sgd
#test_1_run()

#El segundo test, se realizara con 8 neuronas hidden, con una funcion de error de 
#mean_squared_error y un optimizador de sgd
#test_2_run()

#El tercer test, se realizara con 8 neuronas hidden, con una funcion de error de 
#binary_crossentropy y un optimizador de adam
#test_3_run()

#El cuarto test, se realizara con 8 neuronas hidden, con una funcion de error de 
#mean_squared_error y un optimizador de adam
test_4_run()