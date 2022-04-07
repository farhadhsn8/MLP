from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from MLP import MLP


iris = datasets.load_iris()
features = iris.data  
target = pd.get_dummies(iris.target).to_numpy()
features.shape   # (150, 4)
target.shape

dataset = np.hstack(( features,target,np.reshape(iris.target,(-1,1))))
#---------------shuffle---------------------
from sklearn.utils import shuffle
dataset=shuffle(dataset)

#-------------test & train ---------------
train=dataset[0:120,:]    
test=dataset[120:,:]  
test.shape                #(30, 7)
train.shape              # (120, 7)


PARAMS = {
    # enter learning rate :0.1
    # enter code of function for layer0 =>[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ] :4
    # enter number of Perceptrons for  layer 1 (start layer number from 0) : 2
    # enter code of function for layer1 =>[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ] :4
    # enter code of function for layer2 =>[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ] :4
    'LEARNING_RATE' : 0.01 ,
    'CODE_OF_ACTIVATION_FUNCTIONS' : [4,1,3] , #[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ]
    'NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS' : [5]
  }

PARAMS['NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS']


IRIS_MLP = MLP(train[:,0:4] ,train[:,4:7] ,PARAMS )
# IRIS_MLP = MLP(train[:,0:4] ,5 * np.reshape(train[:,7],(-1,1,1)) ,PARAMS)
# IRIS_MLP = MLP(np.array([[1,1]]) ,np.array([[2,2]]) , PARAMS )
# IRIS_MLP = MLP(s1 ,s2, PARAMS )


IRIS_MLP.trainingModel(100)


s=0
k=0
for i in test: #test or train
  est = IRIS_MLP.calculateMLP_outputForRow_k(i[0:4])
  print(est , i[4:7])
  k+=int(np.argmax(est) == np.argmax(i[4:7]) )
  s+=1

print(k, s , str(k/s * 100)+'%')