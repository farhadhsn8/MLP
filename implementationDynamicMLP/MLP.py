import math
import numpy as np
import random


class MLP:

#   PARAMS = {
#     # enter learning rate :0.1
#     # enter code of function for layer0 =>[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ] :4
#     # enter number of Perceptrons for  layer 1 (start layer number from 0) : 2
#     # enter code of function for layer1 =>[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ] :4
#     # enter code of function for layer2 =>[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ] :4

#     'LEARNING_RATE' : 0.1 ,
#     'CODE_OF_ACTIVATION_FUNCTIONS' : [4,4,4] ,
#     'NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS' : [2]
#  }

  def __init__(self , feacturesOftrainingdata , lablesOftrainingdata , parameters):
    self.parameters = parameters
    self.feacturesOftrainingdata = feacturesOftrainingdata
    self.lablesOftrainingdata = lablesOftrainingdata
    self.slidingHead = 0
    self.etha = self.parameters['LEARNING_RATE']
    self.numberOfLayers = len(self.parameters['CODE_OF_ACTIVATION_FUNCTIONS'])
    self.layers =  np.empty(self.numberOfLayers,dtype=Layer)
    self.makeLayers()
    
    

  def makeLayers(self):
    for i in range(self.numberOfLayers):
      self.layers[i] = Layer(i , self)

  def train(self):
    return self.calculateMLP_outputForRow_k(self.getCurrentFeatureRow())


  def calculateMLP_outputForRow_k(self, X):
    self.resetAllcaches()
    return self.layers[self.numberOfLayers - 1].claculateLayerOutput(X)


  def getCurrentFeatureRow(self):
    return self.feacturesOftrainingdata[self.slidingHead]

  def getCurrentLableRow(self):
    return self.lablesOftrainingdata[self.slidingHead]


  def updateAllWeightsByBackPropagationAlgorithm(self):
    for layer in self.layers[:0:-1]:
      layer.updateLayerWeights( False)
    for layer in self.layers[:0:-1]:
      layer.updateLayerWeights( True)
    self.resetAllcaches()
    

  
  def trainingModel(self, epoch=1):
    
    for i in range(epoch):
      # printProgressBar(i, epoch, prefix = 'Progress:', suffix = 'Complete', length = 50)
      self.slidingHead =0
      for j in range(self.feacturesOftrainingdata.shape[0]):
        self.updateAllWeightsByBackPropagationAlgorithm()
        self.slidingHead +=1
      # printProgressBar(i + 1, epoch, prefix = 'Progress:', suffix = 'Complete', length = 50)

  def resetAllcaches(self):
    for i in range(self.numberOfLayers):
      self.layers[i].resetOutput()
      for j in range(self.layers[i].numberOfPerceptrons):
        self.layers[i].perceptrons[j].resetDelta()




  def clearAll(self):
      for i in range(self.numberOfLayers):
        self.layers[i].resetOutput()
        for j in range(self.layers[i].numberOfPerceptrons):
          self.layers[i].perceptrons[j].resetDelta()
          for k in range(self.layers[i].perceptrons[j].numberOfInputs):
            self.layers[i].perceptrons[j].inputBranchs[k].setW()
            print(self.layers[i].perceptrons[j].inputBranchs[k].w)
          

      

  



    



#--------------------------------------------------------------------------


class Layer:

  def __init__(self,layerAddress , MLP):
    self.MLP = MLP
    self.layerAddress = layerAddress
    self.numberOfPerceptrons = self.setNumberOfPerceptrons()
    self.activityFunction = ActivityFunction(self)
    self.perceptrons =  np.empty(self.numberOfPerceptrons,dtype=Perceptron)
    self.perceptrons = self.makePerceptrons()
    self.output = np.full((self.numberOfPerceptrons), math.inf)


  def resetOutput(self):
    self.output = np.full((self.numberOfPerceptrons), math.inf)


  def setNumberOfPerceptrons(self):
    if(self.layerAddress == 0 ):
      return  self.MLP.feacturesOftrainingdata.shape[1]
    if(self.layerAddress == self.MLP.numberOfLayers - 1 ):
      return  self.MLP.lablesOftrainingdata.shape[1]
    return self.MLP.parameters['NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS'][self.layerAddress-1]

  def makePerceptrons(self):
    perceptrons =  np.empty(self.numberOfPerceptrons,dtype=Perceptron) 
    for i in range( self.numberOfPerceptrons ):
      perceptrons[i] = Perceptron( i , self)
    return perceptrons

  def getPreviosLayer(self):
    return self.layerAddress != 0 and self.MLP.layers[self.layerAddress - 1 ] or -1

  
  def getNextLayer(self):
    return self.layerAddress != self.MLP.numberOfLayers - 1 \
     and self.MLP.layers[self.layerAddress + 1 ] or -1


  def claculateLayerOutput(self,X):     # receive Vector   # return Vector

    if ((any(self.output==math.inf))==False):
      return self.output
    if(self.layerAddress==0):
      X = X
      return X
    else:
      X = self.getPreviosLayer().claculateLayerOutput(X)
    output =  np.empty(self.numberOfPerceptrons)
    for i in range(self.numberOfPerceptrons):
      if(self.layerAddress == 0 ):
        output[i] = X[i]
      else:
        output[i] = self.perceptrons[i].calculatePerceptronOutput(X)
    self.output = output
    return self.output

  def calculateDerivativeOfActivationFunction(self,net):
    return self.activityFunction.calculateDerivative(net)

  def updateLayerWeights(self, hardUpdate = False):
    for perceptron in self.perceptrons:
      perceptron.updatePerceptronWeights(hardUpdate)
    

  



#--------------------------------------------------------------------------





class ActivityFunction:
  
  def __init__(self,layer):
    self.layer = layer
    self.functionType = self.layer.MLP.parameters['CODE_OF_ACTIVATION_FUNCTIONS'][self.layer.layerAddress]
  
  def runActivationFunction(self,x):
    if (self.functionType == 1) :
      return self.sigmoid(x)
    if (self.functionType == 2) :
      return self.tanh(x)
    if (self.functionType == 3) :
      return self.ReLU(x)
    if (self.functionType == 4) :
      return self.linear(x)

  def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))

  def tanh(self , x):
    t=(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    return t

  def ReLU(self ,x):
    return max(0.0,x)

  def linear(self , x):
    return x

  def calculateDerivative(self , net):
    if (self.functionType == 1) :
      sig = self.sigmoid(net)
      return (1-sig)*sig
    if (self.functionType == 2) :
      return 1 - self.tanh(net)**2
    if (self.functionType == 3) :
      if(net<0):
        return 0
      return 1
    if (self.functionType == 4) :
      return 1



#--------------------------------------------------------------------------


class Perceptron:

  

  def __init__(self , perceptronNumber , layer ):   # [layerAddress  ,  perceptron] 
    self.baias = 0  # 0 or 1
    self.perceptronNumber = perceptronNumber
    self.layer = layer
    self.numberOfInputs  =  self.getNumberOfInputs()
    self.inputBranchs =  np.empty(self.numberOfInputs,dtype=Layer)
    self.makeInputs()
    self.delta = math.inf

  def resetDelta(self):
    self.delta = math.inf

  def makeInputs(self):
    for i in range(self.numberOfInputs):
      self.inputBranchs[i] = InputBranch(self , i)

  def getNumberOfInputs(self):
    if(self.layer.layerAddress == 0 ):
      return  1
    return self.layer.getPreviosLayer().numberOfPerceptrons + self.baias #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  return self.layer.getPreviosLayer().numberOfPerceptrons + 1 

  def calculatePerceptronOutput(self , X):
        net = self.calculatePerceptronNet(X)
        return self.layer.activityFunction.runActivationFunction(net)

        
  def calculatePerceptronNet(self , X):    # X is input feature vector
        y=0
        # DONT FORGET BAIAS
        X = np.concatenate((X, [self.baias]), axis=None)  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   X = np.concatenate((X, [1]), axis=None) 
        for i in range(self.numberOfInputs):
          y += self.inputBranchs[i].calculateBranchOutput(X[i])
        return y


  def getDelta(self):
    if(self.delta != math.inf):
      return self.delta
    desiredOutput=0
    if(self.layer.layerAddress==self.layer.MLP.numberOfLayers - 1):
      desiredOutput = self.layer.MLP.getCurrentLableRow()[self.perceptronNumber]
    X = (self.layer.layerAddress == 0)  and self.layer.MLP.getCurrentFeatureRow() or self.layer.getPreviosLayer().claculateLayerOutput(self.layer.MLP.getCurrentFeatureRow())
    self.delta =  self.calculateDelta(X ,desiredOutput)
    return self.delta
    # print(self.layer.layerAddress, self.perceptronNumber,self.delta)

  def calculateDelta(self,X , desiredOutput):  # X is input vector 
    net = self.calculatePerceptronNet(X)
    if(self.layer.layerAddress == self.layer.MLP.numberOfLayers - 1):     # perceptron in output layer
      return self.layer.calculateDerivativeOfActivationFunction(net) * ( desiredOutput - self.calculatePerceptronOutput(X))
    else:       # perceptron in hidden layer
      sigma = 0
      # layerOutput = self.layer.claculateLayerOutput(self.layer.MLP.getCurrentFeatureRow())
      for perceptron in self.layer.getNextLayer().perceptrons:
        sigma += (perceptron.inputBranchs[self.perceptronNumber].w * perceptron.getDelta()) 
      return self.layer.calculateDerivativeOfActivationFunction(net) * sigma

  
  def updatePerceptronWeights(self,hardUpdate = False):
    for inputBranch in self.inputBranchs:
      hardUpdate and inputBranch.updateW() or inputBranch.updateWnew()


    


  
#--------------------------------------------------------------------------

class InputBranch:
  
  def __init__(self , perceptron, inputNumber):
    self.inputNumber = inputNumber
    self.perceptron = perceptron
    self.setW()
    self.Wnew = self.w

  def setW(self):
    if(self.perceptron.layer.layerAddress == 0):
      self.w =  1
    self.w = random.uniform(0,1) #00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

  def calculateBranchOutput(self , x):
    return self.w * x 

  

  def updateWnew(self):
    etha = self.perceptron.layer.MLP.etha
    yi = np.concatenate((self.perceptron.layer.getPreviosLayer().claculateLayerOutput(self.perceptron.layer.MLP.getCurrentFeatureRow()), [self.perceptron.baias]), axis=None)[self.inputNumber]  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    self.Wnew =self.w +  etha * self.perceptron.getDelta() * yi 

  def updateW(self):
    self.w = self.Wnew
