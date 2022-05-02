import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit



class RBFNetClassifer():

    def __init__(self , features , targets , numberOfGaussians , eta = 0.01):
        self.features = features
        self.targets = targets
        self.numberOfGaussians = numberOfGaussians
        self.eta = eta
        self.clusters , self.centers  = self.k_means()
        self.stds = self.calculate_std()
        self.bias = np.random.rand()
        self.W = np.random.rand(numberOfGaussians)
       

    def k_means(self):
        random_indices = np.random.choice(self.features.shape[0] , self.numberOfGaussians, replace=False)
        centers = self.features[random_indices]

        converged = False

        clusters = np.zeros(self.features.shape[0]).astype(int)
        while not converged:

            for i in range(self.features.shape[0]):
                min_distance = 999999
                for j in range(centers.shape[0]):
                    if min_distance >  np.linalg.norm(self.features[i] - centers[j]) : 
                        min_distance =  np.linalg.norm(self.features[i] - centers[j])
                        clusters[i] = j
            centers_copy = centers.copy()
            for i in range(centers.shape[0]):
                centers[i] = self.features[clusters==i].mean(axis= 0)
            
            if (centers - centers_copy).sum() < 0.00001 : 
                converged = True
            for i in range(centers.shape[0]):
                centers[i] = self.features[clusters==i].mean(axis= 0)
        return clusters , centers

    def gaussianLayer(self  , x):
      out = []
      for i in range(self.numberOfGaussians):
        d = np.linalg.norm(x - self.centers[i])
        out.append( np.exp(-((d/self.stds)**2)))
        # out.append(np.exp(-1 / (2 * self.stds**2) * (d)**2))

      return np.array(out)

        

    def calculate_std(self):
      max = -1
      for center1 in self.centers :
        for center2 in self.centers :
          if max <  np.linalg.norm(center1 - center2) : 
              max =  np.linalg.norm(center1 - center2)

      return max / ((2 * self.numberOfGaussians)**(1/2))

    def calculateOutput(self , G_layer):
      # print('self.W' ,self.W.shape , 'G_layer ' ,G_layer.shape  )
      y_net = G_layer.dot(self.W) + self.bias
      return expit(y_net)

    def training(self ,  epochs = 5):
      for ep in range(epochs):
        print(ep+1)
        for i in range(self.features.shape[0]):
          G_Layer = self.gaussianLayer(self.features[i])
          y_p = self.calculateOutput(G_Layer)
          # print('self.targets[i]' ,self.targets[i] )
          delta = y_p * (1 - y_p) * (self.targets[i] - y_p)
          # delta = y_p - self.targets[i]
          # print('111111111' ,self.W.shape ,' self.eta ' ,  self.eta  ,'G_Layer' ,G_Layer.shape )
          self.W = self.W + self.eta * delta * G_Layer
          # print('22222222' ,self.W.shape   )
          self.bias = self.bias + self.eta * delta * 1


    def predict(self , newData):
      return self.calculateOutput(self.gaussianLayer(newData))
                    









df = pd.read_csv('Q4.csv')

from sklearn.utils import shuffle
df = shuffle(df)

data_set = np.array(df.values)

train = data_set[:800]
test = data_set[800:]

# X  = df['x'].values.reshape(-1,1)
# Y  = df['y'].values.reshape(-1,1)
# D = df['D'].values

# TT = np.hstack((X,Y))
Z = RBFNetClassifer(train[:,0:2], train[:,2].reshape(-1,1) ,50 , 0.1)

A , _ = Z.k_means()

fig, ax = plt.subplots()

ax.scatter(train[:,0], train[:,1], c=A)
plt.show()



test.shape



Z.training(50)





true = 0
for i in range(train.shape[0]):

  n = Z.predict(train[i , 0:2])
  # print(n ,int(n > 0.5) , D[i])
  if(int(n > 0.5) == train[i , 2]) : 
    true += 1
print(true / train.shape[0])



true = 0
for i in range(test.shape[0]):

  n = Z.predict(test[i , 0:2])
  # print(n ,int(n > 0.5) , D[i])
  if(int(n > 0.5) == test[i , 2]) : 
    true += 1
print(true / test.shape[0])