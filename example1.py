from core.NeuronalNetwork import NeuronalNetwork
from data.Example1DataAgent import Example1DataAgent
from core.NeuronalNetworkLayer import NeuronalNetworkLayer

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
model.train(data.trainDataSet)
print(model.evaluate_model(data.validationDataSet))
#print(model.predict(data.validationDataSet))