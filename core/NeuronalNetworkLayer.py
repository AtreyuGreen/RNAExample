'''
Identifica una capa de la red neuronal
'''
class NeuronalNetworkLayer(object):
    '''
    Constructor
    @param [int] inputNodeCount: Identifica el numero de entradas que tiene esa capa.
    @param [string] activationFunction: Es el nombre de la funcion de activacion.
        sigmoid: Es una funcion de activacion sigmoid
        tahn: Es la funcion de activacion tahn
        relu: Es la funcion de activacion relu
    '''
    def __init__(self, inputNodeCount, activationFunction):
        self.numberNode = inputNodeCount
        self.activationFunction = activationFunction
