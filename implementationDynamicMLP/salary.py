import pandas as pd
from MLP import MLP
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Salary_Data.csv')

# print(df.shape)      #(30, 2)

from sklearn.utils import shuffle
df=shuffle(df)
#-------------test & train ---------------
train=df.iloc[0:24,:]
test=df.iloc[24:,:]




PARAMS = {
    # enter learning rate :0.1
    # enter code of function for layer0 =>[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ] :4
    # enter number of Perceptrons for  layer 1 (start layer number from 0) : 2
    # enter code of function for layer1 =>[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ] :4
    # enter code of function for layer2 =>[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ] :4
    'LEARNING_RATE' : 0.01 ,
    'CODE_OF_ACTIVATION_FUNCTIONS' : [4,4] , #[ 1.sigmoid  | 2.tanh  | 3.relu | 4.linear ]
    'NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS' : []
  }




SALARY_MLP = MLP(train['YearsExperience'].values.reshape(-1,1) ,train['Salary'].values.reshape(-1,1) ,PARAMS )


SALARY_MLP.trainingModel(100)

def plotting(test):
    results = []
    test = test.sort_values(by=['YearsExperience'])
    for i in test.values: #test or train
        est = SALARY_MLP.calculateMLP_outputForRow_k(i[0])
        results.append(est)
    
    print(test.values[: , 0] , test.values[: , 1])


    sns.set_style("darkgrid")
    plt.figure(figsize = (12,4))
    plt.plot(test.values[: , 0], test.values[: , 1] , marker = '*')
    plt.plot(test.values[: , 0], results , marker = '*')
    plt.xlabel("YearsExperience")
    plt.ylabel("Salary")
    plt.legend(["desired", "predict"] , loc='best')
    plt.grid()
    plt.show()

plotting(train)
plotting(test)


