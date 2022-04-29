import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RBFNetClassifer():

    def __init__(self , features , targets , numberOfGaussians):
        self.features = features
        self.targets = targets
        self.numberOfGaussians = numberOfGaussians
        # self.means , self.std = self.k_means()
        # self.k_means()

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
        return clusters


                    


df = pd.read_csv('Q4.csv')

X  = df['x'].values.reshape(-1,1)
Y  = df['y'].values.reshape(-1,1)
TT = np.hstack((X,Y))
Z = RBFNetClassifer(np.array(TT) , np.array([Y]) ,5)

A = Z.k_means()

fig, ax = plt.subplots()




ax.scatter(X.T, Y.T, c=A)

plt.show()
    # def training(self):


    # def predict(self , newData):
