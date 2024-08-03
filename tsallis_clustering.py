
#%%

import copy
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm.auto import tqdm
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

random_seed = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print('WARNING! Using CPU!!!')

# Load the data
dataset_path = 'usdc_weth_005.pickle'
with open(dataset_path, 'rb') as f:
    df = pickle.load(f)
df.index = pd.to_datetime(df.index)

print('Fine prima fase')

# To avoid duplicates
potential_duplicated = df[ df.duplicated() ].block_number.unique()
sure_duplicated = list()
for block in tqdm(potential_duplicated):
    temp_df = df[ df.block_number==block ]
    if (temp_df.duplicated()).sum()*2 == len(temp_df):
        sure_duplicated.append(block)
df = df[~ (df['block_number'].isin(sure_duplicated) & df.duplicated())]

# In this step, I just need the swap data
df = df[ (df.Event == 'Swap_X2Y') | (df.Event == 'Swap_Y2X') ]

# Aggregate the data in df every minute. Keep the last value of each interval
df = df.resample('1T').last()

#%% Clustering algorithms

from sklearn.preprocessing import StandardScaler

class K_Means():
    ''' K-Means'''
    def __init__(self, params=dict()):
        self.set_params(params) #Define parameters of the clustering

    def set_params(self, params):
        '''
        tol = tolerance required to convergence; default=1e-6
        max_it = maximum number of iterations; default=200
        h1 = kernel size; default=35
        h2 = stride; default=7
        seed = set the random seed
        verbose = if true, return info on the convergence
        '''
        self.params = {'tol':1e-6, 'max_it':200, 'h1':35, 'h2':7, 'seed':None, 'verbose':False} #Default parameters
        self.params.update(params) #Update default parameters with those provided by the users

    def fit(self, series):
        self.K = self.slice_data(series) #Slice the return time series
        self.K = self.create_points(self.K) #Create the empirical distributions
        self.init_centroids3clusters() #Initialize centroids and clusters
        self.clustering() #Create the clusters (on the distributions)

    def slice_data(self, series):
        # Slicing of the data
        windows = list() #Define the list of empirical distributions
        start, end = len(series)-self.params['h1'], len(series) #Slice the return time series
        while start >= 0: #We work from the last window to the first one
            windows.append( series[start:end].copy() )
            start -= self.params['h2']
            end -= self.params['h2']
        windows.reverse() #Reverse the array to sort the distributions from the first one to the last
        return np.array(windows)

    def init_centroids3clusters(self):
        # Randomly drawn centroids from K
        self.centroids = list()
        np.random.seed(self.params['seed'])
        temp = np.random.choice(range(len(self.K)), self.params['k'])
        for val in temp:
            self.centroids.append(self.K[val])
        self.clusters = np.zeros(len(self.K)) #Initialize cluster results

    def clustering(self):
        flag, self.it = True, 0
        while flag:
            for n_mu, mu in enumerate(self.K): #Assign closest centroid to each empirical distribution
                closest = np.argmin([self.distance(mu, centr) for centr in self.centroids]) #Compute closest centroid
                self.clusters[n_mu] = closest #Assign to the cluster
            new_centroids = self.update_centroids()  #update centroids
            loss = np.sum([self.distance(old, new) for old, new in zip(self.centroids, new_centroids)]) # Calculate loss_function
            self.it += 1
            if (loss < self.params['tol']) or (self.it >= self.params['max_it']): #Verify exit condition
                flag = False
            self.centroids = new_centroids #Update centroids
        if loss >= self.params['tol']: #Check convergence
            if self.params['verbose']:
                raise ValueError(f'Convergence failed: the maximum number of iterations has been reached. Loss value: {loss}')
        elif self.params['verbose']:
            print(f'Convergence reached after {self.it} steps. Loss value: {loss}')

    def Point2Clusters(self, i, clusters=None):
        if type(clusters) == type(None):
            clusters = self.clusters
        t = i // self.params['h2'] + 1
        ratio = int(self.params['h1'] / self.params['h2'])
        if t <= ratio:
            out = list(range(t))
        elif t <= len(clusters):
            out = list(range(t - ratio, t))
        else:
            out = list(range(t - ratio, len(clusters)))
        return clusters[out]

    def majority_vote(self, a):
        max_votes = 0 #Initialize maximum number of votes
        candidates = np.unique(a) #Obtain candidates
        for can in candidates:
            votes = list(a).count(can)
            if votes > max_votes: #Check if this candidate is winning
                max_votes, win = votes, can
        return win

    def classify(self, clusters=None):
        if type(clusters) == type(None):
            clusters = self.clusters
        pred = list()
        for ret in range( len(clusters)*self.params['h2'] + self.params['h1'] - self.params['h2'] ):
            pred.append(self.majority_vote(self.Point2Clusters(ret, clusters)))
        return np.array(pred)

    def switch_clusters(self, new_order=None):
        if new_order==None:
            self.clusters = 1-self.clusters; self.centroids.reverse()
        else:
            new_centrs, new_clusters = list(), np.zeros(len(self.clusters))
            for cl in range(self.params['k']): #Apply the best permutation
                new_clusters[ self.clusters==cl ] = new_order[cl]
                new_centrs.append( self.centroids[new_order[cl]] )
            self.clusters = new_clusters
            self.centroids = np.array(new_centrs)
            self.classify()

    def predict_update(self, series):
        self.params['verbose'] = False
        #Predict step
        new_wins = self.slice_data(series) #Slice the new data to obtain the new windows
        new_wins = self.create_points(new_wins) #Compute points to cluster from the new slices
        out_clusters = list()
        for mu in new_wins: #Assign closest centroid to each new empirical distribution
            closest = np.argmin([self.distance(mu, centr) for centr in self.centroids]) #Compute closest centroid
            out_clusters.append(closest) #Assign to the cluster
        out_clusters = np.array(out_clusters)
        #Update step
        self.K = np.concatenate([self.K, new_wins])
        self.clusters = np.concatenate([self.clusters, out_clusters])
        self.clustering()
        return out_clusters

    def CountCl(self, a):
        probs = np.zeros(self.params['k'])
        for cl in range(self.params['k']):
            probs[cl] = list(a).count(cl)
        return probs

    def ComputeAccuracies(self, real_regimes, clusters=None):
        regime_acc, acc = list(), np.zeros(2) #Initialize accuracies as vectors [numerator, den]
        for cl in range(self.params['k']): #We need one regime accuracy for each regime
            regime_acc.append( np.zeros(2) )
        for ret, reg in enumerate(real_regimes): #Iterate over each point to compute the accuracies
            count = self.CountCl(self.Point2Clusters(ret, clusters))
            #Update the regime accuracy corresponding to the true class
            regime_acc[reg][0] += count[reg]
            regime_acc[reg][1] += np.sum(count)
            acc[0] += count[reg]; acc[1] += np.sum(count) #Update total accuracy
        #Compute the ultimate accuracies
        acc = acc[0]/acc[1]
        for cl in range(self.params['k']): #We need one regime accuracy for each regime
            regime_acc[cl] = regime_acc[cl][0] / regime_acc[cl][1]
        return acc, regime_acc

    def Davies_Bouldin(self):
      #Compute average distance intra-cluster
      dists = list()
      for cl in range(self.params['k']):
          dists.append(
              np.mean([self.distance(x, self.centroids[cl]) for x in self.K[self.clusters==cl]]))
      #Compute the Davies-Bouldin index
      self.db_index = 0
      for cl in range(self.params['k']): #For every cluster, costructs (d_i+d_j)/distance(c_i, c_j)
          temp_vals = list()
          for cl2 in range(cl):
              temp_vals.append( (dists[cl]+dists[cl2])/self.distance(self.centroids[cl], self.centroids[cl2]) )
          for cl2 in range(cl+1, self.params['k']):
              temp_vals.append( (dists[cl]+dists[cl2])/self.distance(self.centroids[cl], self.centroids[cl2]) )
          self.db_index += np.max(temp_vals) #Only the max value contributes to the index
      self.db_index /= self.params['k']

    def Dunn(self, alpha=1, seed=None):
        if alpha == 1:
            curr_dij, curr_di = np.inf, 0
            for cl in range(self.params['k']):
                #Compute d_{i,j} for every cluster
                for cl2 in range(cl+1, self.params['k']):
                    dijs = list()
                    for x in self.K[self.clusters==cl]:
                        for y in self.K[self.clusters==cl2]:
                            dijs.append(self.distance(x, y))
                    if np.min(dijs) < curr_dij:
                        curr_dij = np.min(dijs)
                #Compute d_i
                dis = list()
                for x_idx, x in enumerate(self.K[self.clusters==cl]):
                    for y in self.K[self.clusters==cl][x_idx:]:
                        dis.append(self.distance(x, y))
                if np.max(dis) > curr_di:
                    curr_di = np.max(dis)
            self.dunn_index = curr_dij/curr_di
        else:
            #Randomly draw the points to work with
            np.random.seed(None)
            indexes = [np.random.choice(
                np.arange(len(self.clusters))[self.clusters==cl],
                int(np.sum(self.clusters==cl)*alpha),
                replace=False) for cl in range(self.params['k'])]

            curr_dij, curr_di = np.inf, 0
            for cl in range(self.params['k']):
                #Compute d_{i,j} for every cluster
                for cl2 in range(cl+1, self.params['k']):
                    dijs = list()
                    for x in self.K[indexes[cl]]:
                        for y in self.K[indexes[cl2]]:
                            dijs.append(self.distance(x, y))
                    if np.min(dijs) < curr_dij:
                        curr_dij = np.min(dijs)
                #Compute d_i
                dis = list()
                for x_idx, x in enumerate(self.K[indexes[cl]]):
                    for y in self.K[indexes[cl]][x_idx:]:
                        dis.append(self.distance(x, y))
                if np.max(dis) > curr_di:
                    curr_di = np.max(dis)
            self.dunn_index = curr_dij/curr_di

    def Silhouette(self, alpha=0.2, seed=None):
        #Randomly draw the points to work with
        np.random.seed(seed)
        indexes = [np.random.choice(
            np.arange(len(self.clusters))[self.clusters==cl],
            int(np.sum(self.clusters==cl)*alpha),
            replace=False) for cl in range(self.params['k'])]
        #Compute distances tensor
        distances = dict()
        for cl1 in range(self.params['k']):
            #Compute distances intra-cluster
            temp_dist = np.zeros((len(indexes[cl1]), len(indexes[cl1])))
            for idx1, val1 in enumerate(indexes[cl1]):
                for idx2, val2 in enumerate(indexes[cl1][idx1+1:]):
                    temp_dist[idx1, idx2] = temp_dist[idx2, idx1] = self.distance(self.K[val1], self.K[val2])
            distances[str([cl1, cl1])] = temp_dist
            #Compute distances inter-cluster
            for cl2 in range(cl1+1, self.params['k']):
                temp_dist = list()
                for val1 in indexes[cl1]:
                    temp_temp_dist = list()
                    for val2 in indexes[cl2]:
                        temp_temp_dist.append( self.distance(self.K[val1], self.K[val2]) )
                    temp_dist.append( temp_temp_dist )
                distances[str([cl1, cl2])] = np.array(temp_dist)
        #Now, compute a, b, and the silhouette for each point
        for cl in range(self.params['k']):
            sil = list()
            for x_idx, x in enumerate(indexes[cl]):
                #Compute a value for every point in the current cluster
                cl_len = len(indexes[cl]) - 1
                a = np.sum(distances[str([cl, cl])][x_idx])/cl_len
                #Now, compute b
                temp_bs = list()
                for cl2 in range(cl):
                    cl_len = len(indexes[cl2])
                    temp_name = [cl, cl2]; temp_name.sort()
                    temp_bs.append( np.sum(distances[str(temp_name)][:,x_idx])/cl_len )
                for cl2 in range(cl+1, self.params['k']):
                    cl_len = len(indexes[cl2])
                    temp_name = [cl, cl2]; temp_name.sort()
                    temp_bs.append( np.sum(distances[str(temp_name)][x_idx])/cl_len )
                b = np.min(temp_bs)
                sil.append( (b-a)/np.max([a,b]) )
        #Mean the silhouettes to obtain the ultimate coefficient
        self.sil_coeff = np.mean(sil)

    def MMD(self, n=1000, sigma=0.1, new_K=False, series=None, seed=None):
        np.random.seed(seed)
        self.mmd = list()
        if new_K:
            temp_K = self.slice_data(series)
            for idx in range(len(temp_K)):
                temp_K[idx].sort()
            temp_K = np.array(temp_K)
            for nnn in range(n):
                temp_mmd = np.zeros((self.params['k'], self.params['k']))
                for cl1 in range(self.params['k']):
                    for cl2 in range(cl1, self.params['k']):
                        X = np.random.choice(np.arange(len(temp_K))[self.clusters==cl1])
                        X = temp_K[X].reshape(-1,1)
                        Y = np.random.choice(np.arange(len(temp_K))[self.clusters==cl2])
                        Y = temp_K[Y].reshape(-1,1)
                        temp_mmd[cl1, cl2] = temp_mmd[cl2, cl1] = np.sqrt(
                            np.mean(rbf_kernel(X, X, sigma)) + np.mean(rbf_kernel(Y, Y, sigma)) - 2*np.mean(rbf_kernel(X, Y, sigma)))
                self.mmd.append(temp_mmd)
        else:
            for nnn in range(n):
                temp_mmd = np.zeros((self.params['k'], self.params['k']))
                for cl1 in range(self.params['k']):
                    for cl2 in range(cl1, self.params['k']):
                        X = np.random.choice(np.arange(len(self.K))[self.clusters==cl1])
                        X = self.K[X].reshape(-1,1)
                        Y = np.random.choice(np.arange(len(self.K))[self.clusters==cl2])
                        Y = self.K[Y].reshape(-1,1)
                        temp_mmd[cl1, cl2] = temp_mmd[cl2, cl1] = np.sqrt(
                            np.mean(rbf_kernel(X, X, sigma)) + np.mean(rbf_kernel(Y, Y, sigma)) - 2*np.mean(rbf_kernel(X, Y, sigma)))
                self.mmd.append(temp_mmd)
        self.mmd = np.array(self.mmd)

class WK_Means(K_Means):
    ''' Wasserstein K-Means'''
    def set_params(self, params):
        '''
        k = number of expected clusters; default=2
        p = order of the Wasserstein distance; default=1
        tol = tolerance required to convergence; default=1e-6
        max_it = maximum number of iterations; default=200
        h1 = kernel size; default=35
        h2 = stride; default=7
        seed = set the random seed
        verbose = if true, return info on the convergence
        '''
        self.params = {'k':2, 'p':1, 'tol':1e-6, 'max_it':200,
                       'h1':35, 'h2':7, 'seed':None, 'verbose':False} #Default parameters
        self.params.update(params) #Update default parameters with those provided by the users

    def distance(self, u, v):
        #For the WK-Means we exploit the Wasserstein distance
        n = len(u); sum = 0 #Initialize variable
        for j in range(n): #Compute the distance
            sum += abs(u[j] - v[j])**self.params['p']
        sum /= n #Normalize by the sample size
        return sum

    def create_points(self, windows):
        #For the WK-Means, the points are the empirical distributions
        points = list()
        for idx in range(len(windows)):
            points.append(windows[idx])
            points[-1].sort()
        return np.array(points)

    def update_centroids(self):
        #Update centroids according to the Wasserstein barycenter
        if self.params['p'] == 1:
            return [np.median(self.K[ np.where(self.clusters==cl) ], axis=0) for cl in range(self.params['k'])]
        else:
            return [np.mean(self.K[ np.where(self.clusters==cl) ], axis=0) for cl in range(self.params['k'])]

    def copy(self):
        copy = WK_Means()
        copy.clusters = self.clusters
        copy.centroids = self.centroids
        copy.params = self.params
        return copy

class MK_Means(K_Means):
    ''' Moment K-Means'''
    def set_params(self, params):
        '''
        k = number of expected clusters; default=2
        p = moments to consider; default=2
        tol = tolerance required to convergence; default=1e-6
        max_it = maximum number of iterations; default=200
        h1 = kernel size; default=35
        h2 = stride; default=7
        seed = set the random seed
        verbose = if true, return info on the convergence
        '''
        self.params = {'k':2, 'p':2, 'tol':1e-6, 'max_it':200,
                       'h1':35, 'h2':7, 'seed':None, 'verbose':False} #Default parameters
        self.params.update(params) #Update default parameters with those provided by the users

    def distance(self, u, v):
        #For the MK-Means we exploit the euclidean distance
        n = len(u); sum = 0 #Initialize variable
        for j in range(n): #Compute the distance
            sum += abs(u[j] - v[j])**2
        sum /= n #Normalize by the sample size
        return sum

    def create_points(self, windows):
        #For the MK-Means, the points are the standardized moment maps
        points = list()
        for idx in range(len(windows)): #Compute the moment maps
            points.append([np.mean(np.array(windows[idx])**(p+1)) for p in range(self.params['p'])])
        points = np.array(points)
        self.scaler = StandardScaler() #Standardize
        return self.scaler.fit_transform(points)

    def update_centroids(self):
        #Update centroids
        return [np.mean(self.K[ np.where(self.clusters==cl) ], axis=0) for cl in range(self.params['k'])]

    def copy(self):
        copy = MK_Means()
        copy.clusters = self.clusters
        copy.centroids = self.centroids
        copy.params = self.params
        return copy

class K_Means_PP(K_Means):
    ''' K-Means ++'''
    def init_centroids3clusters(self):
        self.centroids = list()
        np.random.seed(self.params['seed'])
        self.centroids.append(self.K[np.random.choice(range(len(self.K)))]) #Randomly drawn the first centroid
        for n_centroids in range(1, self.params['k']):
            #Compute distances
            distances = list()
            flag = 0
            for mu in self.K: #Assign closest centroid to each empirical distribution
                distances.append( np.min([self.distance(mu, centr) for centr in self.centroids])**2 )
            distances /= np.sum(distances)
            #Sample the next centroid with probability weighted by the distance
            val = np.random.rand()
            sum = 0
            for n_mu in range(len(self.K)):
                if sum < val:
                    sum += distances[n_mu]
                    if sum >= val:
                        self.centroids.append(self.K[n_mu])
        self.clusters = np.zeros(len(self.K)) #Initialize cluster results

class WK_Means_PP(K_Means_PP, WK_Means):
    ''' Wasserstein K-Means ++'''
    def copy(self):
        copy = WK_Means_PP()
        copy.clusters = self.clusters
        copy.centroids = self.centroids
        copy.params = self.params
        return copy

class MK_Means_PP(K_Means_PP, MK_Means):
    ''' Moment K-Means ++'''
    def copy(self):
        copy = MK_Means_PP()
        copy.clusters = self.clusters
        copy.centroids = self.centroids
        copy.params = self.params
        return copy

#%% Apply the clustering

df.ffill(inplace=True)
df['LogReturn'] = np.log(df.price) - np.log(df.price.shift(1)) #In the first instance, we work with hourly returns
df = df[1:]
start_date = datetime(2021, 7, 1, 00, 00, 00)
end_date = datetime(2024, 5, 1, 00, 00, 00)


print('\033[1mWasserstein\033[0m')
# Hourly windows, 5 minutes stride
wk_pars = {'k':2, 'p':1, 'tol':1e-6, 'max_it':200, 'h1':60, 'h2':5, 'seed':random_seed, 'verbose':True}
df_w = df[(df.index >= start_date) & (df.index < end_date)]
df_w = df_w[len(df_w.LogReturn) % wk_pars['h2']:]
mdl_w = WK_Means_PP(wk_pars); mdl_w.fit(np.array(df_w.LogReturn))
mdl_w_pred = mdl_w.classify()

if len(mdl_w_pred[ mdl_w_pred==0 ])/len(mdl_w_pred) < 0.5: #Class 0 is the biggest one
    mdl_w.switch_clusters(); mdl_w_pred = 1-mdl_w_pred
df_w['Pred'] = [int(p) for p in mdl_w_pred]

print('\n\n')

print('\033[1mMoment\033[0m')
mk_pars = {'k':2, 'p':2, 'tol':1e-6, 'max_it':200, 'h1':60, 'h2':5, 'seed':random_seed, 'verbose':True}
df_m = df[(df.index >= start_date) & (df.index < end_date)]
df_m = df_m[len(df_m.LogReturn) % wk_pars['h2']:]
mdl_m = MK_Means_PP(mk_pars); mdl_m.fit(np.array(df_m.LogReturn))
mdl_m_pred = mdl_m.classify()

if len(mdl_m_pred[ mdl_m_pred==0 ])/len(mdl_m_pred) < 0.5: #Class 0 is the biggest one
    mdl_m.switch_clusters(); mdl_m_pred = 1-mdl_m_pred
df_m['Pred'] = [int(p) for p in mdl_m_pred]

#Map the windows in the mean-variance plane
points = mdl_m.slice_data(np.array(df_m.LogReturn))
points = [[np.std(dist), np.mean(dist)] for dist in points]
points = np.array(points)
labels_w = [f'Class {int(cl)}' for cl in mdl_w.clusters] #Separate windows according their cluster
labels_m = [f'Class {int(cl)}' for cl in mdl_m.clusters]

#Map the centroids in the mean-variance plane
centroids_w = np.array([ [np.std(cl_centr), np.mean(cl_centr)] for cl_centr in mdl_w.centroids ])
centroids_m = mdl_m.scaler.inverse_transform(mdl_m.centroids)
centroids_m[0] = [np.sqrt(centroids_m[0,1]), centroids_m[0,0]]
centroids_m[1] = [np.sqrt(centroids_m[1,1]), centroids_m[1,0]]

#Plot the clusters in the mean-variance plane
fig, ax = plt.subplots(1, 2, figsize=(20,6))

sub_p = 0
sns.scatterplot(x=points[:,0], y=points[:, 1], hue=labels_w, ax=ax[sub_p], s=15)
sns.scatterplot(x=centroids_w[:1,0], y=centroids_w[:1, 1], ax=ax[sub_p], s=300, marker='*', label='Centroid 0', color=sns.color_palette()[2])
sns.scatterplot(x=centroids_w[1:,0], y=centroids_w[1:, 1], ax=ax[sub_p], s=300, marker='*', label='Centroid 1', color=sns.color_palette()[3])
ax[sub_p].set_title('Wasserstein Clustering', fontsize='x-large')
ax[sub_p].set_xlabel('Standard Deviation', fontsize='x-large')
ax[sub_p].set_ylabel('Mean', fontsize='x-large')
ax[sub_p].legend(fontsize='large')

sub_p = 1
sns.scatterplot(x=points[:,0], y=points[:, 1], hue=labels_m, ax=ax[sub_p], s=15)
sns.scatterplot(x=centroids_m[:1,0], y=centroids_m[:1, 1], ax=ax[sub_p], s=300, marker='*', label='Centroid 0', color=sns.color_palette()[2])
sns.scatterplot(x=centroids_m[1:,0], y=centroids_m[1:, 1], ax=ax[sub_p], s=300, marker='*', label='Centroid 1', color=sns.color_palette()[3])
ax[sub_p].set_title('Moment Clustering', fontsize='x-large')
ax[sub_p].set_xlabel('Standard Deviation', fontsize='x-large')
ax[sub_p].set_ylabel('Mean', fontsize='x-large')
ax[sub_p].legend(fontsize='large')

plt.suptitle('USDC/WETH Clustering - Mean-Variance Plane', fontsize='xx-large')
plt.tight_layout()
#plt.savefig('usdc_weth_005_cluster_mean_var.png', dpi=200)
plt.show()

# WARNING: DA ORA FINO ALLA FINE DELLA CELLA C'È IL CODICE PER PLOTTARE
# IL CLUSTERING VISTO DIRETTAMENTE SULLA TIME SERIES (FIGURE 12 DEL REPORT SEB)
# CI METTE PARECCHIO TEMPO PER GIRARE, CIRCA UN'ORETTA E MEZZA CON I DATI WETH-USDC
#Define the colors to use in the plot
def give_color(col):
    if col in [0, 1]:
        return sns.color_palette()[0]
    elif col in [2, 3]:
        return sns.color_palette()[9]
    elif col in [4, 5]:
        return sns.color_palette()[2]
    elif col in [6, 7, 8]:
        return sns.color_palette()[8]
    elif col in [9, 10]:
        return sns.color_palette()[6]
    elif col in [11, 12]:
        return sns.color_palette()[3]

#Define the quantities to plot
x_axis = df_m.index
y_axis = np.array(df_m.price)
m_p, M_p = np.min(df_m.price), np.max(df_m.price)

fig, ax = plt.subplots(2, 1, figsize=(15,12))
max_show = 75

#Plot 1 - Wasserstein
sub_p = 0
label_flags = [True for _ in range(13)] #To handle labels for the legend
old_reg, start_idx = np.sum(mdl_w.Point2Clusters(0), dtype=int), 0 #Save current label and starting index
for ret in tqdm(range(len(df_m.price)), desc='Plotting Wasserstein'):
    new_reg = np.sum(mdl_w.Point2Clusters(ret), dtype=int)
    if old_reg != new_reg:
        if label_flags[old_reg]:
            label_flags[old_reg] = False
            sns.lineplot(x=x_axis[start_idx:ret], y=y_axis[start_idx:ret], ax=ax[sub_p], color=give_color(old_reg), label=f'{int(old_reg*100/12)}% Class 1')
        else:
            sns.lineplot(x=x_axis[start_idx:ret], y=y_axis[start_idx:ret], ax=ax[sub_p], color=give_color(old_reg))
        old_reg, start_idx = new_reg, ret
sns.lineplot(x=x_axis[start_idx:], y=y_axis[start_idx:], ax=ax[sub_p], color=give_color(old_reg))
ax[sub_p].set_xlabel('Time', fontsize='x-large')
ax[sub_p].set_ylabel('Price', fontsize='x-large')
ax[sub_p].set_title('Wasserstein Clustering', fontsize='x-large')
ax[sub_p].legend(fontsize='large', bbox_to_anchor=(1.02, 1.))

#Plot 2 - Moment
sub_p = 1
label_flags = [True for _ in range(13)] #To handle labels for the legend
old_reg, start_idx = np.sum(mdl_m.Point2Clusters(0), dtype=int), 0 #Save current label and starting index
for ret in tqdm(range(len(df_m.price)), desc='Plotting Moment'):
    new_reg = np.sum(mdl_m.Point2Clusters(ret), dtype=int)
    if old_reg != new_reg:
        if label_flags[old_reg]:
            label_flags[old_reg] = False
            sns.lineplot(x=x_axis[start_idx:ret], y=y_axis[start_idx:ret], ax=ax[sub_p], color=give_color(old_reg), label=f'{int(old_reg*100/12)}% Class 1')
        else:
            sns.lineplot(x=x_axis[start_idx:ret], y=y_axis[start_idx:ret], ax=ax[sub_p], color=give_color(old_reg))
        old_reg, start_idx = new_reg, ret
sns.lineplot(x=x_axis[start_idx:], y=y_axis[start_idx:], ax=ax[sub_p], color=give_color(old_reg))
ax[sub_p].set_xlabel('Time', fontsize='x-large')
ax[sub_p].set_ylabel('Price', fontsize='x-large')
ax[sub_p].set_title('Moment Clustering', fontsize='x-large')
ax[sub_p].legend(fontsize='large', bbox_to_anchor=(1.02, 1.))

#Final adjustment
plt.suptitle('USDC-WETH 0.05%', fontsize='xx-large')
plt.tight_layout()
plt.savefig('../figures/usdc_weth_005_cluster_ts.png', dpi=200)
plt.show()

#%% Try to fit the qGaussian distribution

import qGaussian

#--------------------------------------- Fit on the whole dataset
print('Fitting the qGaussian distribution on the whole dataset')
x_train = 100 * df_w.LogReturn.dropna().values.astype(np.float64)
x_train = x_train[ x_train != 0 ] # Remove the zeros

# Count the number of outliers
lq = np.quantile(x_train, 0.25)
uq = np.quantile(x_train, 0.75)
iqr = 1.5 * (uq - lq)
outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
print('There are', outliers_pct, '% outliers in the data')

fitted_values = qGaussian.fit(x_train, n_it=200)
fitted_q, fitted_mu, fitted_sigma =\
    fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
print(f"Fitted q: {fitted_q}, Fitted mu: {fitted_mu}, Fitted sigma: {fitted_sigma}")
print('\n')

#--------------------------------------- Fit on the class 0
print('Fitting the qGaussian distribution on the class 0')
x_train = 100 * df_w[df_w.Pred == 0].LogReturn.dropna().values.astype(np.float64)
x_train = x_train[ x_train != 0 ] # Remove the zeros

# Count the number of outliers
lq = np.quantile(x_train, 0.25)
uq = np.quantile(x_train, 0.75)
iqr = 1.5 * (uq - lq)
outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
print('There are', outliers_pct, '% outliers in the data')

fitted_values = qGaussian.fit(x_train, n_it=200)
fitted_q, fitted_mu, fitted_sigma =\
    fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
print(f"Fitted q: {fitted_q}, Fitted mu: {fitted_mu}, Fitted sigma: {fitted_sigma}")
print('\n')

#--------------------------------------- Fit on the class 1
print('Fitting the qGaussian distribution on the class 1')
x_train = 100 * df_w[df_w.Pred == 1].LogReturn.dropna().values.astype(np.float64)
x_train = x_train[ x_train != 0 ] # Remove the zeros

# Count the number of outliers
lq = np.quantile(x_train, 0.25)
uq = np.quantile(x_train, 0.75)
iqr = 1.5 * (uq - lq)
outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
print('There are', outliers_pct, '% outliers in the data')

fitted_values = qGaussian.fit(x_train, n_it=200)
fitted_q, fitted_mu, fitted_sigma =\
    fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
print(f"Fitted q: {fitted_q}, Fitted mu: {fitted_mu}, Fitted sigma: {fitted_sigma}")
print('\n')

#%% Try to fit the qGaussian distribution - Daily fit

import qGaussian

#--------------------------------------- Fit on the whole dataset
q_values_list = list()
sigma_list = list()
mean_list = list()
outliers_pct_list = list()
total_obs_list = list()
error_list = list()

for day in tqdm(pd.date_range(datetime(2021, 7, 1, 00, 00, 00),
                              datetime(2024, 5, 1, 00, 00, 00),
                              freq='M',
                              inclusive='left'),
                desc='Iterating over the days'):
    start_date = datetime(day.year, day.month, day.day, 00, 00, 00)
    end_day = day + pd.DateOffset(months=1)
    end_date = datetime(end_day.year, end_day.month, end_day.day, 00, 00, 00)
    x_train = 100 * df_w[(df_w.index >= start_date) & (df_w.index < end_date)].LogReturn.dropna().values.astype(np.float64)
    x_train = x_train[ x_train != 0] # Remove the zeros
    total_obs_list.append(len(x_train)) # Add the total number of observations

    # Count the number of outliers
    lq = np.quantile(x_train, 0.25)
    uq = np.quantile(x_train, 0.75)
    iqr = 1.5 * (uq - lq)
    outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
    outliers_pct_list.append(outliers_pct) # Add the percentage of outliers
    outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

    try:
        fitted_values = qGaussian.fit(x_train, n_it=200)
        fitted_q, fitted_mu, fitted_sigma =\
            fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
        q_values_list.append(fitted_q) # Add the fitted q value
        mean_list.append(fitted_mu) # Add the mean value
        sigma_list.append(fitted_sigma) # Add the fitted sigma value
    except:
        print('Error in the fitting process!')
        error_list.append(day)

whole_res = [total_obs_list, mean_list, outliers_pct_list,
             q_values_list, sigma_list, error_list]

#--------------------------------------- Fit on the class 0
q_values_list = list()
sigma_list = list()
mean_list = list()
outliers_pct_list = list()
total_obs_list = list()
error_list = list()

for day in tqdm(pd.date_range(datetime(2021, 7, 1, 00, 00, 00),
                              datetime(2024, 5, 1, 00, 00, 00),
                              freq='M',
                              inclusive='left'),
                desc='Iterating over the days'):
    start_date = datetime(day.year, day.month, day.day, 00, 00, 00)
    end_day = day + pd.DateOffset(months=1)
    end_date = datetime(end_day.year, end_day.month, end_day.day, 00, 00, 00)
    x_train = 100 * df_w[(df_w.index >= start_date) & (df_w.index < end_date) & (df_w.Pred == 0)].LogReturn.dropna().values.astype(np.float64)
    x_train = x_train[ x_train != 0] # Remove the zeros
    total_obs_list.append(len(x_train)) # Add the total number of observations

    # Count the number of outliers
    lq = np.quantile(x_train, 0.25)
    uq = np.quantile(x_train, 0.75)
    iqr = 1.5 * (uq - lq)
    outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
    outliers_pct_list.append(outliers_pct) # Add the percentage of outliers
    outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

    try:
        fitted_values = qGaussian.fit(x_train, n_it=200)
        fitted_q, fitted_mu, fitted_sigma =\
            fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
        q_values_list.append(fitted_q) # Add the fitted q value
        mean_list.append(fitted_mu) # Add the mean value
        sigma_list.append(fitted_sigma) # Add the fitted sigma value
    except:
        print('Error in the fitting process!')
        error_list.append(day)

class0_res = [total_obs_list, mean_list, outliers_pct_list,
             q_values_list, sigma_list, error_list]

#--------------------------------------- Fit on the class 1
q_values_list = list()
sigma_list = list()
mean_list = list()
outliers_pct_list = list()
total_obs_list = list()
error_list = list()

for day in tqdm(pd.date_range(datetime(2021, 7, 1, 00, 00, 00),
                              datetime(2024, 5, 1, 00, 00, 00),
                              freq='M',
                              inclusive='left'),
                desc='Iterating over the days'):
    start_date = datetime(day.year, day.month, day.day, 00, 00, 00)
    end_day = day + pd.DateOffset(months=1)
    end_date = datetime(end_day.year, end_day.month, end_day.day, 00, 00, 00)
    x_train = 100 * df_w[(df_w.index >= start_date) & (df_w.index < end_date) & (df_w.Pred == 1)].LogReturn.dropna().values.astype(np.float64)
    x_train = x_train[ x_train != 0] # Remove the zeros
    total_obs_list.append(len(x_train)) # Add the total number of observations

    # Count the number of outliers
    lq = np.quantile(x_train, 0.25)
    uq = np.quantile(x_train, 0.75)
    iqr = 1.5 * (uq - lq)
    outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
    outliers_pct_list.append(outliers_pct) # Add the percentage of outliers
    outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

    try:
        fitted_values = qGaussian.fit(x_train, n_it=200)
        fitted_q, fitted_mu, fitted_sigma =\
            fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
        q_values_list.append(fitted_q) # Add the fitted q value
        mean_list.append(fitted_mu) # Add the mean value
        sigma_list.append(fitted_sigma) # Add the fitted sigma value
    except:
        print('Error in the fitting process!')
        error_list.append(day)

class1_res = [total_obs_list, mean_list, outliers_pct_list,
             q_values_list, sigma_list, error_list]

#--------------------------------------- Comparative Clustering
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
for n_to_plot, to_plot in enumerate(['Mean Value', 'Outliers %', 'q Value', 'sigma Value']):
    if n_to_plot == 3:
        sns.scatterplot(x=range(len(whole_res[n_to_plot+1])),
                        y=whole_res[n_to_plot+1], label='Total',
                        ax=ax[n_to_plot//2, n_to_plot%2])
        sns.scatterplot(x=range(len(class0_res[n_to_plot+1])),
                        y=class0_res[n_to_plot+1], label='Class 0',
                        ax=ax[n_to_plot//2, n_to_plot%2])
        sns.scatterplot(x=range(len(class1_res[n_to_plot+1])),
                        y=class1_res[n_to_plot+1], label='Class 1',
                        ax=ax[n_to_plot//2, n_to_plot%2])
        ax[n_to_plot//2, n_to_plot%2].axhline(np.mean(whole_res[n_to_plot+1]), color=sns.color_palette()[0],
                label='Mean Tot')
        ax[n_to_plot//2, n_to_plot%2].axhline(np.mean(class0_res[n_to_plot+1]), color=sns.color_palette()[1],
                label='Mean C0')
        ax[n_to_plot//2, n_to_plot%2].axhline(np.mean(class1_res[n_to_plot+1]), color=sns.color_palette()[2],
                label='Mean C1')
        plt.legend(bbox_to_anchor=(1, 1.05))
    else:
        sns.scatterplot(x=range(len(whole_res[n_to_plot+1])),
                        y=whole_res[n_to_plot+1],
                        ax=ax[n_to_plot//2, n_to_plot%2])
        sns.scatterplot(x=range(len(class0_res[n_to_plot+1])),
                        y=class0_res[n_to_plot+1],
                        ax=ax[n_to_plot//2, n_to_plot%2])
        sns.scatterplot(x=range(len(class1_res[n_to_plot+1])),
                        y=class1_res[n_to_plot+1],
                        ax=ax[n_to_plot//2, n_to_plot%2])
        ax[n_to_plot//2, n_to_plot%2].axhline(np.mean(whole_res[n_to_plot+1]), color=sns.color_palette()[0])
        ax[n_to_plot//2, n_to_plot%2].axhline(np.mean(class0_res[n_to_plot+1]), color=sns.color_palette()[1])
        ax[n_to_plot//2, n_to_plot%2].axhline(np.mean(class1_res[n_to_plot+1]), color=sns.color_palette()[2])

    ax[n_to_plot//2, n_to_plot%2].set_title(to_plot)
plt.suptitle("Comparison of the clusters' statistical properties")
plt.tight_layout()
#plt.savefig('../figures/return_clustering_q_gauss.png', dpi=150)
plt.show()

#%% Unconditional NeuralCDF

import wandb

# Define the base neural network class
class BaseNN():
    '''
    Base class for neural networks.
    '''
    def __init__(self):
        '''
        Initialize the base neural network class.
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
        self.set_regularizer() # Define the regularization
    
    def set_regularizer(self):
        '''
        Set the regularization according to the parameter 'reg_type' (default: no regularization).
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
        #Two available regularization types: l1, l2, and l1_l2
        self.reg = torch.tensor(self.params['reg']).to(self.dev)
        #Eventually, apply l1 regularization
        if self.params['reg_type'] == 'l1':
            def l1_model_reg(model):
                regularization_loss = torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    regularization_loss += torch.norm(param, 1)
                return self.reg*regularization_loss
            self.regularizer = l1_model_reg
        #Eventually, apply l2 regularization
        elif self.params['reg_type'] == 'l2':
            def l2_model_reg(model):
                regularization_loss = torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    regularization_loss += torch.norm(param, 2)
                return self.reg*regularization_loss
            self.regularizer = l2_model_reg
        #Eventually, apply l1_l2 regularization
        elif self.params['reg_type'] == 'l1_l2':
            def l1_l2_model_reg(model):
                l1_loss, l2_loss = torch.tensor(0.0).to(self.dev), torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    l1_loss += torch.norm(param, 1)
                    l2_loss += torch.norm(param, 2)
                return self.reg[0]*l1_loss + self.reg[1]*l2_loss
            self.regularizer = l1_l2_model_reg
        #Eventually, no regularization is applied
        else:
            def no_reg(model):
                return torch.tensor(0.0)
            self.regularizer = no_reg
        
    def set_optimizer(self):
        '''
        Set the optimizer according to the parameter 'optimizer' (default: Adam).
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
        #Two available optimizers: Adam, RMSprop
        temp = self.params['optimizer'].lower() if type(self.params['optimizer']) == str else None
        if temp.lower() == 'adam':
            from torch.optim import Adam
            self.opt = Adam(self.mdl.parameters(), self.params['lr'])
        elif temp.lower() == 'rmsprop':
            from torch.optim import RMSprop
            self.opt = RMSprop(self.mdl.parameters(), self.params['lr'])
        else:
            raise ValueError(f"Optimizer {temp} not recognized")
        
    def early_stopping(self, curr_loss):
        '''
        Early stopping function.
        INPUT:
            - curr_loss: float,
                current loss.
        OUTPUT:
            - output: bool,
                True if early stopping is satisfied, False otherwise.
        '''
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss #Save the best loss
            self.best_model = copy.deepcopy(self.mdl.state_dict()) #Save the best model
            self.patience = 0 #Reset the patience counter
        else:
            self.patience += 1
        #Check if I have to exit
        if self.patience > self.params['patience']:
            output = True
        else:
            output = False
        return output

    def plot_losses(self, yscale='log'):
        '''
        Plot the training loss and, eventually, the validation loss.
        INPUT:
            - yscale: str, optional
                scale of the y-axis. Default is 'log'.
        OUTPUT:
            - None.
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme()
        #Plot the losses
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        ax.set(yscale=yscale)
        sns.lineplot(self.train_loss, label='Train')
        if isinstance(self.val_loss, list):
            sns.lineplot(self.val_loss, label='Validation')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Losses")
        plt.legend()
        plt.show()

# Define the feedforward neural network class
class FFN(nn.Module):
    '''
    Class for Feedforward neural networks.
    '''
    def __init__(self, layers, init, activation, drop, out_act='linear', DTYPE=torch.float64):
        '''
        INPUT:
            - layers: list of int
                list such that each component is the number of neurons in the corresponding layer.
            - init: str
                the type of initialization to use for the weights. Either 'glorot_normal' or 'glorot_uniform'.
            - activation: str
                name of the activation function. Either 'tanh', 'relu', or 'sigmoid'.
            - drop: float
                dropout rate.
            - DTYPE: torch data type.
        OUTPUT:
            - None.
        '''
        super().__init__()
        # Define the layers
        self.layers = nn.ModuleList([
            nn.Linear(layers[l-1], layers[l], dtype=DTYPE) for\
                l in range(1,len(layers))
                ])
        # Initialize the weights
        self.weights_initializer(init)
        #Define activation function and dropout
        self.set_activation_function(activation, out_act) #Define activation function
        self.dropout = nn.Dropout(drop)
    
    def weights_initializer(self, init):
        '''
        Initialize the weights.
        INPUT:
            - init: str
                type of initialization to use for the weights.
        OUTPUT:
            - None.
        '''
        #Two available initializers: glorot_normal, glorot_uniform
        temp = init.lower() if type(init) == str else None
        if temp == 'glorot_normal':
            for layer in self.layers:
                nn.init.xavier_normal_(layer.weight)
        elif temp == 'glorot_uniform':
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight)
        else:
            raise ValueError(f"Initializer {init} not recognized")

    def set_activation_function(self, act, out_act):
        '''
        Set the activation function.
        INPUT:
            - activation: str
                type of activation function to use.
        OUTPUT:
            - None.
        '''
        #Four available activation functions: tanh, relu, sigmoid, linear (that is, identity)
        acts = list()
        for activation in [act, out_act]:
            if activation == 'tanh':
                acts.append( nn.Tanh() )
            elif activation == 'relu':
                acts.append( nn.ReLU() )
            elif activation == 'sigmoid':
                acts.append( nn.Sigmoid() )
            elif activation == 'softplus':
                acts.append( nn.Softplus() )
            elif activation == 'linear':
                acts.append( nn.Identity() )
            else:
                raise ValueError(f"Activation function {activation} not recognized")
        self.activation = acts[0]; self.out_act = acts[1]
        
    def forward(self, x):
        '''
        INPUT:
            - x: torch.Tensor
                input of the network; shape (batch_size, n_features).
        OUTPUT:
            - output: torch.Tensor
                output of the network; shape (batch_size, output_size).
        '''
        # Forward pass through the network
        for layer in self.layers[:-1]:
            x = self.activation(layer(x)) #Hidden layers
            x = self.dropout(x) #Dropout
        output = self.out_act(self.layers[-1](x)) #Output layer
        return output

class CRPS(nn.Module):
    '''
    Class for the CRPS loss function.
    '''
    def __init__(self, n_points=500):
        '''
        Initialize the CRPS loss function.
        INPUT:
            - quantiles: list of float
                each element is between 0 and 1 and represents a target confidence levels.
        OUTPUT:
            - None.
        '''
        super().__init__()
        self.n_points = n_points
    
    def forward(self, y_pred, y_true, indicator=False):
        '''
        INPUT:
            - y_pred: torch.Tensor
                quantile forecasts with shape (batch_size, n_series).
            - y_true: torch.Tensor
                actual values with shape (batch_size, n_series).
        OUTPUT:
            - loss: float
                mean CRPS loss.
        '''
        # Ensure to work with torch tensors
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        #Check consistency in the dimensions
        if len(y_pred.shape) == 1:
            y_pred = torch.unsqueeze(y_pred, dim=1)
        if len(y_true.shape) == 1:
            y_true = torch.unsqueeze(y_true, dim=1)
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(f'Shape[0] of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) do not match!!!')
        if y_pred.shape[1] != len(self.quantiles):
            raise ValueError(f'Shape[1] of y_pred ({y_pred.shape}) and len(quantiles) ({len(self.quantiles)}) do not match!!!')
        if y_true.shape[1] != 1:
            raise ValueError(f'Shape[1] of y_true ({y_pred.shape}) should be 1!!!')
        # Compute the pinball loss
        error = y_true - y_pred
        loss = torch.zeros(y_true.shape).to(y_true.device)
        for q, quantile in enumerate(self.quantiles):
            loss += torch.max(quantile * error[:,q:q+1], (quantile - 1) * error[:,q:q+1])
        loss = torch.mean(loss)
        return loss

# Define the NeuralCDF class
class NeuralCDF(BaseNN):
    '''
    Expected Shortfall estimation via Kratz approach with Quantile Regression Neural Network
        (QRNN) for quantile regression.
    '''
    def __init__(self, params, dev, verbose=True):
        '''
        Initialization of the K-QRNN model.
        INPUTS:
            - params: dict
                parameters of the model.
            - dev: torch.device
                indicates the device where the model will be trained.
            - verbose: bool, optional
                if True, print the training progress. Default is True.
        PARAMS:
            - optimizer: str, optional
                optimizer to use, either 'Adam' or 'RMSProp'. Default is 'Adam'.
            - reg_type: str or None, optional
                type of regularization. Either None, 'l1', 'l2', or 'l1_l2'. Default is None.
            - reg: float or list of float, optional
                regularization parameter. Not consider when reg_type=None.
                float when reg_type='l1' or 'l2'. List with two floats (l1 and l2) when
                reg_type='l1_l2'. Default is 0.
            - initializer: str, optional
                initializer for the weights. Either 'glorot_normal' or 'glorot_uniform'.
                Default is 'glorot_normal'.
            - activation: str, optional
                activation function. Either 'relu', 'sigmoid', or 'tanh'. Default is 'relu'.
            - lr: float, optional
                learning rate. Default is 0.01.
            - dropout: float, optional
                dropout rate. Default is 0.
            - batch_size: int, optional
                batch size. Default is -1, that is full batch. When
                batch_size < x_train.shape[0], mini-batch training is performed.
            - patience: int, optional
                patience for early stopping. Default is np.inf, that is no early stopping.
            - verbose: int, optional
                set after how many epochs the information on losses are printed. Default is 1.
        OUTPUTS:
            - None.
        '''
        self.set_params(params) #Set the parameters
        self.dev = dev
        self.verbose = verbose
        super().__init__()
        # Define the model and optimizer
        self.mdl = FFN(self.params['layers'], self.params['initializer'],
                       self.params['activation'], self.params['dropout'],
                       out_act=self.params['out_activation']).to(self.dev)
        self.set_optimizer() #Define the optimizer
    
    def loss_fast(self, y_pred, y_true):
        loss = torch.tensor(0, dtype=torch.float64).to(self.dev)
        for i in range(y_true.shape[0]):
            # if i==0:
            #     print('In the loss, y_pred shape:', y_pred.shape, 'y_true shape:', y_true[i].shape)
            loss += nn.functional.mse_loss(y_pred, y_true[i])
        return loss/y_true.shape[0]

    def loss_mem_save(self, y_pred, y_true, z_points):
        loss = torch.tensor(0, dtype=torch.float64).to(self.dev)
        for i in range(y_true.shape[0]):
            loss += nn.functional.mse_loss(y_pred, torch.where(y_true[i] <= z_points, 1, 0))
        return loss/y_true.shape[0]

    def set_params(self, params):
        '''
        Define the ultimate parameters dictionary by merging the parameters
            defined by the user with the default ones
        INPUT:
            - params: dict
                parameters defined by the user.
        OUTPUT:
            - None.
        '''
        self.params = {'optimizer': 'Adam', 'reg_type': None, 'out_activation':'linear',
                       'crps_std_scale': 3, 'crps_points': 500,
                       'reg': 0, 'initializer': 'glorot_normal', 'activation': 'relu',
                       'lr': 0.01, 'dropout': 0, 'batch_size':-1,
                       'patience': np.inf, 'verbose': 1, 'pdf_constr':False} #Default parameters
        self.params.update(params) #Update default parameters with those provided by the user
    
    def train_single_batch(self, z_train, y_train):
        '''
        Training with full batch.
        INPUT:
            - x_train: torch.Tensor
                model's input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's output of shape (batch_size, 1).
        OUTPUT:
            - None.
        '''
        self.mdl.train() #Set the model in training mode
        self.opt.zero_grad() # Zero the gradients
        #print(z_train.shape)
        outputs = self.mdl(z_train) # Forward pass
        #print(outputs.shape)
        #print(y_train.shape)
        loss = self.loss(outputs, y_train) + self.regularizer(self.mdl)
        if self.params['pdf_constr']:
            pdf = torch.autograd.grad(outputs, z_train,
                                      torch.ones_like(outputs), retain_graph=True)[0]
            loss += torch.mean(nn.ReLU()(-pdf))
        loss.backward()  # Backpropagation
        self.opt.step()  # Update weights
        self.train_loss.append(loss.item()) #Save the training loss
    
    def train_multi_batch(self, z_train, y_train, indices):
        '''
        Training with multiple batches.
        INPUT:
            - x_train: torch.Tensor
                model's input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's output of shape (batch_size, 1).
            - indices: torch.Tensor
                list of indices (range(batch_size)).
        OUTPUT:
            - None.
        '''
        self.mdl.train() #Set the model in training mode
        #Prepare batch training
        total_loss = 0.0
        indices = indices[torch.randperm(indices.size(0))] #Shuffle the indices
        # Training
        for i in range(0, len(indices), self.params['batch_size']):
            #Construct the batch
            batch_indices = indices[i:i+self.params['batch_size']] #Select the indices
            y_batch = y_train[batch_indices]

            self.opt.zero_grad() # Zero the gradients
            #print(z_train.shape)
            outputs = self.mdl(z_train) # Forward pass
            #print(outputs.shape)
            #print(y_batch.shape)
            loss = self.loss(outputs, y_batch) + self.regularizer(self.mdl)
            if self.params['pdf_constr']:
                pdf = torch.autograd.grad(outputs, z_train,
                                        torch.ones_like(outputs), retain_graph=True)[0]
                loss += torch.mean(nn.ReLU()(-pdf))
            loss.backward()  # Backpropagation
            self.opt.step()  # Update weights
            total_loss += loss.item()
        total_loss /= np.ceil(len(indices) / self.params['batch_size'])
        self.train_loss.append(total_loss) #Save the training loss

    def fit(self, y_train_, y_val_=None, mem_save=False, wandb_save=False):
        '''
        Fit the model. With early stopping and batch training.
        INPUT:
            - x_train: torch.Tensor
                model's train input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's train output of shape (batch_size, 1).
            - x_val: torch.Tensor, optional
                model's validation input of shape (batch_size, n_features). Default is None.
            - y_val: torch.Tensor, optional
                model's validation output of shape (batch_size, 1). Default is None.
        OUTPUT:
            - None.
        '''
        #Initialize the best model
        self.best_model = self.mdl.state_dict()
        self.best_loss = np.inf
        #Initialize the patience counter
        self.patience = 0
        #Initialize the training and validation losses
        self.train_loss = []
        if isinstance(y_val_, torch.Tensor):
            self.val_loss = []
        else:
            self.val_loss = None
        # Set the wandb environment if required:
        if wandb_save:
            wandb.init(project=f'clustering_unconditional_crps_c{TARGET_CLASS}',
                        config=self.params)
            wandb.watch([self.mdl], log='all')
        #Understand if I'm using single or multiple batches
        single_batch = (self.params['batch_size'] == -1) or\
            (self.params['batch_size'] >= y_train_.shape[0])
        # Eventually, create the train indices list
        if not single_batch: indices = torch.arange(y_train_.shape[0])
        # Drawn the points where evaluate the CRPS
        adj_std = torch.std(y_train_) * self.params['crps_std_scale']
        z_train = torch.linspace(
            -adj_std, adj_std, self.params['crps_points'], dtype=torch.float64,
            requires_grad = self.params['pdf_constr'] ).unsqueeze(dim=1).to(self.dev)
        if isinstance(y_val_, torch.Tensor):
            z_val = torch.distributions.uniform.Uniform(-adj_std, adj_std).sample(
                (self.params['crps_points'],)).unsqueeze(dim=1).requires_grad_(
                    requires_grad = self.params['pdf_constr']).to(self.dev)
        # If not mem_save, compute the indicator function for the train and val once at all
        if not mem_save:
            y_train = torch.concat(
                    [torch.where(val <= z_train, 1., 0) for val in y_train_],
                    dim=1).T.to(self.dev).unsqueeze(dim=2).type(torch.float64)
            if isinstance(y_val_, torch.Tensor):
                y_val = torch.concat(
                    [torch.where(val <= z_val, 1., 0).to(self.dev) for val in y_val_],
                    dim=1
                ).T.unsqueeze(dim=2)
            self.loss = self.loss_fast
        else: #Otherwise, use keep the original tensors
            y_train = y_train_
            y_val = y_val_
            self.loss = self.loss_mem_save
        # Set the verbosity if it is provided as a boolean:
        if isinstance(self.verbose, bool):
            self.verbose = int(self.verbose) - 1
        #Train the model
        if self.verbose >= 0: #If verbose is True, then use the progress bar
            from tqdm.auto import tqdm
            it_base = tqdm(range(self.params['n_epochs']), desc='Training the network') #Create the progress bar
        else: #Otherwise, use the standard iterator
            it_base = range(self.params['n_epochs']) #Create the iterator
        for epoch in it_base:
            if (epoch==0) and (self.verbose > 0):
                print_base = '{:<10}{:<15}{:<15}'
                print(print_base.format('Epoch', 'Train Loss', 'Val Loss'))
            # Training
            if single_batch:
                self.train_single_batch(z_train, y_train)
            else:
                self.train_multi_batch(z_train, y_train, indices)
            # If Validation
            if isinstance(y_val_, torch.Tensor):
                self.mdl.eval() #Set the model in evaluation mode
                if self.params['pdf_constr']:
                        # Compute the validation loss
                        val_output = self.mdl(z_val)
                        if mem_save:
                            val_loss = self.loss(val_output, y_val, z_val)
                        else:
                            val_loss = self.loss(val_output, y_val)
                        pdf = torch.autograd.grad(val_output, z_val,
                                                  torch.ones_like(val_output), retain_graph=True)[0]
                        val_loss += torch.mean(nn.ReLU()(-pdf))
                        self.val_loss.append(val_loss.item()) #Save the validation loss
                else:
                    with torch.no_grad():
                        # Compute the validation loss
                        if mem_save:
                            val_loss = self.loss(self.mdl(z_val), y_val, z_val)
                        else:
                            val_loss = self.loss(self.mdl(z_val), y_val)
                        self.val_loss.append(val_loss.item()) #Save the validation loss
                # Update best model and eventually early stopping
                if self.early_stopping(val_loss):
                    if self.verbose >= 0: print(f"Early stopping at epoch {epoch+1}")
                    break
            
            #Eventually, wandb save
            if wandb_save:
                to_wandb = dict()
                to_wandb[f'train_loss'] = self.train_loss[-1]
                to_wandb[f'val_loss'] = self.val_loss[-1]
                wandb.log(to_wandb)
            else: #Otherwise
                # Update best model and eventually early stopping
                if self.early_stopping(self.train_loss[-1]):
                    if self.verbose >= 0: print(f"Early stopping at epoch {epoch+1}")
                    break
            # Eventually, print the losses
            if self.verbose > 0:
                if (epoch+1) % self.verbose == 0:
                    self.print_losses(epoch+1)
        #After the training, load the best model and return its losses
        self.final_ops(z_train, y_train, z_val, y_val, mem_save)
        #Eventually, wandb save
        if wandb_save:
            to_wandb = dict()
            to_wandb[f'train_loss'] = self.train_loss[-1]
            to_wandb[f'val_loss'] = self.val_loss[-1]
            wandb.log(to_wandb)
        #Eventually, finish the wandb run
        if wandb_save:
            wandb.finish()

    def print_losses(self, epoch):
        print('{:<10}{:<15}{:<15}'.format(
            epoch, format(self.train_loss[-1], '.20f')[:10],
            format(self.val_loss[-1], '.20f')[:10] if not isinstance(
                self.val_loss, type(None)) else '-'))

    def final_ops(self, z_train, y_train, z_val, y_val, mem_save):
        # Recover the best model
        self.mdl.load_state_dict(self.best_model)

        # Compute the train loss of the best model
        z_train.requires_grad_(True)
        output = self.mdl(z_train)
        if mem_save:
            train_loss = self.loss(output, y_train, z_train)
        else:
            train_loss = self.loss(output, y_train)
        pdf = torch.autograd.grad(output, z_train, torch.ones_like(output))[0]
        train_loss += torch.mean(nn.ReLU()(-pdf))
        self.train_loss.append(train_loss.item())

        # Compute the val loss of the best model
        z_val.requires_grad_(True)
        val_output = self.mdl(z_val)
        if mem_save:
            val_loss = self.loss(val_output, y_val, z_val)
        else:
            val_loss = self.loss(val_output, y_val)
        pdf = torch.autograd.grad(val_output, z_val, torch.ones_like(val_output))[0]
        val_loss += torch.mean(nn.ReLU()(-pdf))
        self.val_loss.append(val_loss.item())

        # Print the loss
        print('\n')
        self.print_losses('Final')
    
    def __call__(self, x_test):
        '''
        Predict the quantile forecast and the expected shortfall.
        INPUT:
            - x_test: torch.Tensor
                input of the model; shape (batch_size, n_features).
        OUTPUT:
            - qf: ndarray
             quantile forecast of the model.
            - ef: ndarray
                expected shortfall predicted by the model.
        '''
        return self.mdl(x_test)
    
    def draw(self, n_points, z_min=-1.5, z_max=1.5, N_grid=10_000, seed=None):
        if not isinstance(seed, type(None)):
            torch.manual_seed(seed) #Set the seed
            torch.cuda.manual_seed_all(seed)
        uniform_samples = torch.rand(n_points, device=self.dev) #Sample from a uniform distribution
        
        # Create a grid of z values
        z_values = torch.linspace(z_min, z_max, N_grid, device=self.dev, dtype=torch.float64)
        z_values = z_values.unsqueeze(1)  # Make it (num_points, 1)
        
        # Evaluate the CDF for each z in the grid
        cdf_values = self.mdl(z_values).squeeze(1)  # Make it (num_points,)
        
        # Function to find the z corresponding to each uniform sample using binary search
        def find_z(cdf_value):
            low, high = 0, N_grid - 1
            while low < high:
                mid = (low + high) // 2
                if cdf_values[mid] < cdf_value:
                    low = mid + 1
                else:
                    high = mid
            return z_values[low].item()
        
        # Apply the binary search to each uniform sample
        sampled_points = torch.tensor([find_z(u) for u in uniform_samples], device=self.dev)
        
        return sampled_points

#%% Unconditional - Define the network input and target

del(df)

# Dataset Class 0
log_ret = 100 * df_w[ df_w.Pred == 0 ].LogReturn.dropna().values
log_ret = pd.Series(log_ret, index=df_w[ df_w.Pred == 0 ].index)
x_data = log_ret[(log_ret.index >= end_date-pd.Timedelta(days=42)) &\
                 (log_ret.index < end_date)].astype(np.float64)

y_train0 = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=42)) &\
            (log_ret.index < end_date-pd.Timedelta(days=14))].values.astype(np.float64)).to(device)
y_train0 = y_train0[ y_train0 != 0] # Remove the zeros
y_val0 = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=14)) &\
            (log_ret.index < end_date-pd.Timedelta(days=7))].values.astype(np.float64)).to(device)
y_val0 = y_val0[ y_val0 != 0] # Remove the zeros
y_test0_ = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=7)) &\
            (log_ret.index < end_date)].values.astype(np.float64)).to(device)
y_test0_ = y_test0_[ y_test0_ != 0 ] # Remove the zeros
z_test0 = torch.linspace(-torch.std(y_train0), torch.std(y_train0), 1000, dtype=torch.float64,
                        requires_grad=True ).unsqueeze(dim=1).to(device)
y_test0 = torch.concat(
    [torch.where(val <= z_test0, 1., 0) for val in y_test0_], dim=1
    ).T.to(device).unsqueeze(dim=2).type(torch.float64)

# Dataset Class 1
log_ret = 100 * df_w[ df_w.Pred == 1 ].LogReturn.dropna().values
log_ret = pd.Series(log_ret, index=df_w[ df_w.Pred == 1 ].index)
x_data = log_ret[(log_ret.index >= end_date-pd.Timedelta(days=42)) &\
                 (log_ret.index < end_date)].astype(np.float64)

y_train1 = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=42)) &\
            (log_ret.index < end_date-pd.Timedelta(days=14))].values.astype(np.float64)).to(device)
y_train1 = y_train1[ y_train1 != 0] # Remove the zeros
y_val1 = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=14)) &\
            (log_ret.index < end_date-pd.Timedelta(days=7))].values.astype(np.float64)).to(device)
y_val1 = y_val1[ y_val1 != 0] # Remove the zeros
y_test1_ = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=7)) &\
            (log_ret.index < end_date)].values.astype(np.float64)).to(device)
y_test1_ = y_test1_[ y_test1_ != 0 ] # Remove the zeros
z_test1 = torch.linspace(-torch.std(y_train1), torch.std(y_train1), 1000, dtype=torch.float64,
                        requires_grad=True ).unsqueeze(dim=1).to(device)
y_test1 = torch.concat(
    [torch.where(val <= z_test1, 1., 0) for val in y_test1_], dim=1
    ).T.to(device).unsqueeze(dim=2).type(torch.float64)

#%% Unconditional - Test the best models

# Define the optimal hyperparameters (actually, I've found params0 = params1)
params0 = {'activation':'tanh', 'reg_type':None, 'batch_size':-1, 'dropout':0.2,
            'initializer':'glorot_normal', 'layers':[1, 10, 10, 10, 1], 'lr':5e-4,
            'optimizer':'rmsprop', 'out_activation':'sigmoid', 'n_epochs':3_000,
            'pdf_constr':False, 'crps_points':150, 'crps_std_scale':3, 'patience':20}

params1 = {'activation':'tanh', 'reg_type':None, 'batch_size':-1, 'dropout':0.2,
            'initializer':'glorot_normal', 'layers':[1, 10, 10, 10, 1], 'lr':5e-4,
            'optimizer':'rmsprop', 'out_activation':'sigmoid', 'n_epochs':3_000,
            'pdf_constr':False, 'crps_points':150, 'crps_std_scale':3, 'patience':20}

# Create and train the models
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
mdl0 = NeuralCDF(params0, device, verbose=20)
mdl0.fit(y_train0, y_val0, wandb_save=False)

torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
mdl1 = NeuralCDF(params1, device, verbose=20)
mdl1.fit(y_train1, y_val1, wandb_save=False)

# Plot CDF and PDF
z_plot0 = torch.linspace(-torch.std(y_train0)*3, torch.std(y_train0)*3, 1000, dtype=torch.float64,
                        requires_grad=True ).unsqueeze(dim=1).to(device)
z_plot1 = torch.linspace(-torch.std(y_train1)*3, torch.std(y_train1)*3, 1000, dtype=torch.float64,
                        requires_grad=True ).unsqueeze(dim=1).to(device)
# Compute the CDF
F_z_batch0 = mdl0(z_plot0)
f_z_batch0 = torch.autograd.grad(F_z_batch0, z_plot0, grad_outputs=torch.ones_like(F_z_batch0))[0]
F_z_batch1 = mdl1(z_plot0)
f_z_batch1 = torch.autograd.grad(F_z_batch1, z_plot0, grad_outputs=torch.ones_like(F_z_batch1))[0]
# sort F_z_batch
F_z_batch0, _ = torch.sort(F_z_batch0)
f_z_batch0, _ = torch.sort(f_z_batch0)
F_z_batch1, _ = torch.sort(F_z_batch1)
f_z_batch1, _ = torch.sort(f_z_batch1)
# convert to numpy both z_batch and F_z_batch
F_z_batch0 = F_z_batch0.cpu().detach().numpy()
f_z_batch0 = f_z_batch0.cpu().detach().numpy()
z_batch0 = z_plot0.cpu().detach().numpy()
F_z_batch1 = F_z_batch1.cpu().detach().numpy()
f_z_batch1 = f_z_batch1.cpu().detach().numpy()
z_batch1 = z_plot0.cpu().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# plot the CDF
sub_p = 0
sns.lineplot(x=z_batch0.flatten(), y=F_z_batch0.flatten(), ax=ax[sub_p], label='Class 0')
sns.lineplot(x=z_batch1.flatten(), y=F_z_batch1.flatten(), ax=ax[sub_p], label='Class 1')
ax[sub_p].legend()

# plot the pdf
sub_p = 1
sns.lineplot(x=z_batch0.flatten(), y=f_z_batch0.flatten(), ax=ax[sub_p], label='Class 0')
sns.lineplot(x=z_batch1.flatten(), y=f_z_batch1.flatten(), ax=ax[sub_p], label='Class 1')
ax[sub_p].legend()

plt.suptitle('Unconditional Neural CDF and PDF - Clusters Comparison')
#plt.savefig('../figures/neural_cdf_uncond_cluster.png', dpi=150)
plt.show()

#%% Unconditional - Other metrics and q-q plots

gen0 = mdl0.draw(len(y_train0), z_min=torch.min(y_train0).item(),
                z_max=torch.max(y_train0).item(), seed=random_seed).cpu().detach().numpy()

gen1 = mdl1.draw(len(y_train1), z_min=torch.min(y_train1).item(),
                z_max=torch.max(y_train1).item(), seed=random_seed).cpu().detach().numpy()

# Comparison between stats
print('{:<18}{:<2}{:^30}{:<15}{:<2}{:^30}'.format('', '|', 'Class 0', '', '|', 'Class 1'))
print('{:<18}{:<2}{:<15}{:<30}{:<2}{:<15}{:<30}'.format('Statistic', '|', 'True', 'Generated', '|', 'True', 'Generated'))
print('{:<100}'.format('-'*100))
print('{:<18}{:<2}{:<15}{:<30}{:<2}{:<15}{:<30}'.format('Quantile 0.1', '|', round(torch.quantile(y_train0, 0.1).item(), 5), round(np.quantile(gen0, 0.1), 5), '|', round(torch.quantile(y_train1, 0.1).item(), 5), round(np.quantile(gen1, 0.1), 5)))
print('{:<18}{:<2}{:<15}{:<30}{:<2}{:<15}{:<30}'.format('Quantile 0.25', '|', round(torch.quantile(y_train0, 0.25).item(), 5), round(np.quantile(gen0, 0.25), 5), '|', round(torch.quantile(y_train1, 0.25).item(), 5), round(np.quantile(gen1, 0.25), 5)))
print('{:<18}{:<2}{:<15}{:<30}{:<2}{:<15}{:<30}'.format('Mean', '|', round(torch.mean(y_train0).item(), 5), round(np.mean(gen0), 5), '|', round(torch.mean(y_train1).item(), 5), round(np.mean(gen1), 5)))
print('{:<18}{:<2}{:<15}{:<30}{:<2}{:<15}{:<30}'.format('Median', '|', round(torch.median(y_train0).item(), 5), round(np.median(gen0), 5), '|', round(torch.median(y_train1).item(), 5), round(np.median(gen1), 5)))
print('{:<18}{:<2}{:<15}{:<30}{:<2}{:<15}{:<30}'.format('Quantile 0.75', '|', round(torch.quantile(y_train0, 0.75).item(), 5), round(np.quantile(gen0, 0.75), 5), '|', round(torch.quantile(y_train1, 0.75).item(), 5), round(np.quantile(gen1, 0.75), 5)))
print('{:<18}{:<2}{:<15}{:<30}{:<2}{:<15}{:<30}'.format('Quantile 0.9', '|', round(torch.quantile(y_train0, 0.9).item(), 5), round(np.quantile(gen0, 0.9), 5), '|', round(torch.quantile(y_train1, 0.9).item(), 5), round(np.quantile(gen1, 0.9), 5)))

y_train_sorted0 = np.sort(y_train0.detach().cpu().numpy())
y_train_sorted1 = np.sort(y_train1.detach().cpu().numpy())
gen_sorted0 = np.sort(gen0)
gen_sorted1 = np.sort(gen1)

fig, ax = plt.subplots(2, 1, figsize=(15, 6))

c_p = 0
ax[c_p].plot(y_train_sorted0, gen_sorted0, 'o', markersize=5)
ax[c_p].plot([min(y_train_sorted0), max(y_train_sorted0)], [min(y_train_sorted0), max(y_train_sorted0)], 'r--')
# Customize the plot
ax[c_p].set_xlabel('Quantiles of y_train')
ax[c_p].set_ylabel('Quantiles of Uncond. gen.')
ax[c_p].set_title('Q-Q Plot - Class 0')

c_p = 1
ax[c_p].plot(y_train_sorted1, gen_sorted1, 'o', markersize=5)
ax[c_p].plot([min(y_train_sorted1), max(y_train_sorted1)], [min(y_train_sorted1), max(y_train_sorted1)], 'r--')
# Customize the plot
ax[c_p].set_xlabel('Quantiles of y_train')
ax[c_p].set_ylabel('Quantiles of Cond. gen.')
ax[c_p].set_title('Q-Q Plot - Class 1')

plt.tight_layout()
#plt.savefig('../figures/q_q_plot_clusters.png', dpi=150)
plt.show()

#%% Conditional NeuralCDF

import wandb

# Define the base neural network class
class BaseNN():
    '''
    Base class for neural networks.
    '''
    def __init__(self):
        '''
        Initialize the base neural network class.
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
        self.set_regularizer() # Define the regularization
    
    def set_regularizer(self):
        '''
        Set the regularization according to the parameter 'reg_type' (default: no regularization).
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
        #Two available regularization types: l1, l2, and l1_l2
        self.reg = torch.tensor(self.params['reg']).to(self.dev)
        #Eventually, apply l1 regularization
        if self.params['reg_type'] == 'l1':
            def l1_model_reg(model):
                regularization_loss = torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    regularization_loss += torch.norm(param, 1)
                return self.reg*regularization_loss
            self.regularizer = l1_model_reg
        #Eventually, apply l2 regularization
        elif self.params['reg_type'] == 'l2':
            def l2_model_reg(model):
                regularization_loss = torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    regularization_loss += torch.norm(param, 2)
                return self.reg*regularization_loss
            self.regularizer = l2_model_reg
        #Eventually, apply l1_l2 regularization
        elif self.params['reg_type'] == 'l1_l2':
            def l1_l2_model_reg(model):
                l1_loss, l2_loss = torch.tensor(0.0).to(self.dev), torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    l1_loss += torch.norm(param, 1)
                    l2_loss += torch.norm(param, 2)
                return self.reg[0]*l1_loss + self.reg[1]*l2_loss
            self.regularizer = l1_l2_model_reg
        #Eventually, no regularization is applied
        else:
            def no_reg(model):
                return torch.tensor(0.0)
            self.regularizer = no_reg
        
    def set_optimizer(self):
        '''
        Set the optimizer according to the parameter 'optimizer' (default: Adam).
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
        #Two available optimizers: Adam, RMSprop
        temp = self.params['optimizer'].lower() if type(self.params['optimizer']) == str else None
        if temp.lower() == 'adam':
            from torch.optim import Adam
            self.opt = Adam(self.mdl.parameters(), self.params['lr'])
        elif temp.lower() == 'rmsprop':
            from torch.optim import RMSprop
            self.opt = RMSprop(self.mdl.parameters(), self.params['lr'])
        else:
            raise ValueError(f"Optimizer {temp} not recognized")
        
    def early_stopping(self, curr_loss):
        '''
        Early stopping function.
        INPUT:
            - curr_loss: float,
                current loss.
        OUTPUT:
            - output: bool,
                True if early stopping is satisfied, False otherwise.
        '''
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss #Save the best loss
            self.best_model = copy.deepcopy(self.mdl.state_dict()) #Save the best model
            self.patience = 0 #Reset the patience counter
        else:
            self.patience += 1
        #Check if I have to exit
        if self.patience > self.params['patience']:
            output = True
        else:
            output = False
        return output

    def plot_losses(self, yscale='log'):
        '''
        Plot the training loss and, eventually, the validation loss.
        INPUT:
            - yscale: str, optional
                scale of the y-axis. Default is 'log'.
        OUTPUT:
            - None.
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme()
        #Plot the losses
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        ax.set(yscale=yscale)
        sns.lineplot(self.train_loss, label='Train')
        if isinstance(self.val_loss, list):
            sns.lineplot(self.val_loss, label='Validation')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Losses")
        plt.legend()
        plt.show()

# Define the feedforward neural network class
class FFN(nn.Module):
    '''
    Class for Feedforward neural networks.
    '''
    def __init__(self, layers, init, activation, drop, out_act='linear', DTYPE=torch.float64):
        '''
        INPUT:
            - layers: list of int
                list such that each component is the number of neurons in the corresponding layer.
            - init: str
                the type of initialization to use for the weights. Either 'glorot_normal' or 'glorot_uniform'.
            - activation: str
                name of the activation function. Either 'tanh', 'relu', or 'sigmoid'.
            - drop: float
                dropout rate.
            - DTYPE: torch data type.
        OUTPUT:
            - None.
        '''
        super().__init__()
        # Define the layers
        self.layers = nn.ModuleList([
            nn.Linear(layers[l-1], layers[l], dtype=DTYPE) for\
                l in range(1,len(layers))
                ])
        # Initialize the weights
        self.weights_initializer(init)
        #Define activation function and dropout
        self.set_activation_function(activation, out_act) #Define activation function
        self.dropout = nn.Dropout(drop)
    
    def weights_initializer(self, init):
        '''
        Initialize the weights.
        INPUT:
            - init: str
                type of initialization to use for the weights.
        OUTPUT:
            - None.
        '''
        #Two available initializers: glorot_normal, glorot_uniform
        temp = init.lower() if type(init) == str else None
        if temp == 'glorot_normal':
            for layer in self.layers:
                nn.init.xavier_normal_(layer.weight)
        elif temp == 'glorot_uniform':
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight)
        else:
            raise ValueError(f"Initializer {init} not recognized")

    def set_activation_function(self, act, out_act):
        '''
        Set the activation function.
        INPUT:
            - activation: str
                type of activation function to use.
        OUTPUT:
            - None.
        '''
        #Four available activation functions: tanh, relu, sigmoid, linear (that is, identity)
        acts = list()
        for activation in [act, out_act]:
            if activation == 'tanh':
                acts.append( nn.Tanh() )
            elif activation == 'relu':
                acts.append( nn.ReLU() )
            elif activation == 'swish':
                acts.append( nn.SiLU() )
            elif activation == 'sigmoid':
                acts.append( nn.Sigmoid() )
            elif activation == 'softplus':
                acts.append( nn.Softplus() )
            elif activation == 'linear':
                acts.append( nn.Identity() )
            else:
                raise ValueError(f"Activation function {activation} not recognized")
        self.activation = acts[0]; self.out_act = acts[1]
        
    def forward(self, x):
        '''
        INPUT:
            - x: torch.Tensor
                input of the network; shape (batch_size, n_features).
        OUTPUT:
            - output: torch.Tensor
                output of the network; shape (batch_size, output_size).
        '''
        # Forward pass through the network
        for layer in self.layers[:-1]:
            x = self.activation(layer(x)) #Hidden layers
            x = self.dropout(x) #Dropout
        output = self.out_act(self.layers[-1](x)) #Output layer
        return output

class CRPS(nn.Module):
    '''
    Class for the CRPS loss function.
    '''
    def __init__(self, n_points=500):
        '''
        Initialize the CRPS loss function.
        INPUT:
            - quantiles: list of float
                each element is between 0 and 1 and represents a target confidence levels.
        OUTPUT:
            - None.
        '''
        super().__init__()
        self.n_points = n_points
    
    def forward(self, y_pred, y_true, indicator=False):
        '''
        INPUT:
            - y_pred: torch.Tensor
                quantile forecasts with shape (batch_size, n_series).
            - y_true: torch.Tensor
                actual values with shape (batch_size, n_series).
        OUTPUT:
            - loss: float
                mean CRPS loss.
        '''
        # Ensure to work with torch tensors
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        #Check consistency in the dimensions
        if len(y_pred.shape) == 1:
            y_pred = torch.unsqueeze(y_pred, dim=1)
        if len(y_true.shape) == 1:
            y_true = torch.unsqueeze(y_true, dim=1)
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(f'Shape[0] of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) do not match!!!')
        if y_pred.shape[1] != len(self.quantiles):
            raise ValueError(f'Shape[1] of y_pred ({y_pred.shape}) and len(quantiles) ({len(self.quantiles)}) do not match!!!')
        if y_true.shape[1] != 1:
            raise ValueError(f'Shape[1] of y_true ({y_pred.shape}) should be 1!!!')
        # Compute the pinball loss
        error = y_true - y_pred
        loss = torch.zeros(y_true.shape).to(y_true.device)
        for q, quantile in enumerate(self.quantiles):
            loss += torch.max(quantile * error[:,q:q+1], (quantile - 1) * error[:,q:q+1])
        loss = torch.mean(loss)
        return loss

# Define the NeuralCDF class
class NeuralCondCDF(BaseNN):
    '''
    Expected Shortfall estimation via Kratz approach with Quantile Regression Neural Network
        (QRNN) for quantile regression.
    '''
    def __init__(self, params, dev, verbose=True):
        '''
        Initialization of the K-QRNN model.
        INPUTS:
            - params: dict
                parameters of the model.
            - dev: torch.device
                indicates the device where the model will be trained.
            - verbose: bool, optional
                if True, print the training progress. Default is True.
        PARAMS:
            - optimizer: str, optional
                optimizer to use, either 'Adam' or 'RMSProp'. Default is 'Adam'.
            - reg_type: str or None, optional
                type of regularization. Either None, 'l1', 'l2', or 'l1_l2'. Default is None.
            - reg: float or list of float, optional
                regularization parameter. Not consider when reg_type=None.
                float when reg_type='l1' or 'l2'. List with two floats (l1 and l2) when
                reg_type='l1_l2'. Default is 0.
            - initializer: str, optional
                initializer for the weights. Either 'glorot_normal' or 'glorot_uniform'.
                Default is 'glorot_normal'.
            - activation: str, optional
                activation function. Either 'relu', 'sigmoid', or 'tanh'. Default is 'relu'.
            - lr: float, optional
                learning rate. Default is 0.01.
            - dropout: float, optional
                dropout rate. Default is 0.
            - batch_size: int, optional
                batch size. Default is -1, that is full batch. When
                batch_size < x_train.shape[0], mini-batch training is performed.
            - patience: int, optional
                patience for early stopping. Default is np.inf, that is no early stopping.
            - verbose: int, optional
                set after how many epochs the information on losses are printed. Default is 1.
        OUTPUTS:
            - None.
        '''
        self.set_params(params) #Set the parameters
        self.dev = dev
        self.verbose = verbose
        super().__init__()
        # Define the model and optimizer
        self.mdl = FFN(self.params['layers'], self.params['initializer'],
                       self.params['activation'], self.params['dropout'],
                       out_act=self.params['out_activation']).to(self.dev)
        self.set_optimizer() #Define the optimizer
    
    def loss_fast(self, y_pred, y_true):
        loss = torch.tensor(0, dtype=torch.float64).to(self.dev)
        for i in range(y_true.shape[0]):
            # if i==0:
            #     print('In the loss, y_pred shape:', y_pred.shape, 'y_true shape:', y_true[i].shape)
            loss += nn.functional.mse_loss(y_pred, y_true[i])
        return loss/y_true.shape[0]

    def loss_mem_save(self, y_pred, y_true, z_points):
        loss = torch.tensor(0, dtype=torch.float64).to(self.dev)
        for i in range(y_true.shape[0]):
            loss += nn.functional.mse_loss(y_pred, torch.where(y_true[i] <= z_points, 1, 0))
        return loss/y_true.shape[0]

    def set_params(self, params):
        '''
        Define the ultimate parameters dictionary by merging the parameters
            defined by the user with the default ones
        INPUT:
            - params: dict
                parameters defined by the user.
        OUTPUT:
            - None.
        '''
        self.params = {'optimizer': 'Adam', 'reg_type': None, 'out_activation':'linear',
                       'crps_std_scale': 3, 'crps_points': 500,
                       'reg': 0, 'initializer': 'glorot_normal', 'activation': 'relu',
                       'lr': 0.01, 'dropout': 0, 'batch_size':-1,
                       'patience': np.inf, 'verbose': 1, 'pdf_constr':False} #Default parameters
        self.params.update(params) #Update default parameters with those provided by the user
    
    def train_single_batch(self, z_train, x_train, y_train):
        '''
        Training with full batch.
        INPUT:
            - x_train: torch.Tensor
                model's input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's output of shape (batch_size, 1).
        OUTPUT:
            - None.
        '''
        self.mdl.train() #Set the model in training mode
        self.opt.zero_grad() # Zero the gradients
        #print(z_train.shape)
        loss = 0
        for n_val, val in enumerate(x_train):
            outputs = self.mdl(
                torch.cat((z_train,
                           val.unsqueeze(dim=0).repeat(self.params['crps_points'],1)), dim=1)) # Forward pass
            loss += self.loss(outputs, y_train[n_val:n_val+1])
            if self.params['pdf_constr']:
                pdf = torch.autograd.grad(outputs, z_train,
                                        torch.ones_like(outputs), retain_graph=True)[0]
                loss += torch.mean(nn.ReLU()(-pdf))
        #print(outputs.shape)
        #print(y_train.shape)
        loss /= x_train.shape[0] #Average the loss
        loss += self.regularizer(self.mdl) #Add the regularization
        loss.backward()  # Backpropagation
        self.opt.step()  # Update weights
        self.train_loss.append(loss.item()) #Save the training loss
    
    def train_multi_batch(self, z_train, y_train, indices):
        '''
        Training with multiple batches.
        INPUT:
            - x_train: torch.Tensor
                model's input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's output of shape (batch_size, 1).
            - indices: torch.Tensor
                list of indices (range(batch_size)).
        OUTPUT:
            - None.
        '''
        self.mdl.train() #Set the model in training mode
        #Prepare batch training
        total_loss = 0.0
        indices = indices[torch.randperm(indices.size(0))] #Shuffle the indices
        # Training
        for i in range(0, len(indices), self.params['batch_size']):
            #Construct the batch
            batch_indices = indices[i:i+self.params['batch_size']] #Select the indices
            y_batch = y_train[batch_indices]

            self.opt.zero_grad() # Zero the gradients
            #print(z_train.shape)
            outputs = self.mdl(z_train) # Forward pass
            #print(outputs.shape)
            #print(y_batch.shape)
            loss = self.loss(outputs, y_batch) + self.regularizer(self.mdl)
            if self.params['pdf_constr']:
                pdf = torch.autograd.grad(outputs, z_train,
                                        torch.ones_like(outputs), retain_graph=True)[0]
                loss += torch.mean(nn.ReLU()(-pdf))
            loss.backward()  # Backpropagation
            self.opt.step()  # Update weights
            total_loss += loss.item()
        total_loss /= np.ceil(len(indices) / self.params['batch_size'])
        self.train_loss.append(total_loss) #Save the training loss

    def fit(self, x_train, y_train_, x_val=None, y_val_=None, mem_save=False, wandb_save=False):
        '''
        Fit the model. With early stopping and batch training.
        INPUT:
            - x_train: torch.Tensor
                model's train input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's train output of shape (batch_size, 1).
            - x_val: torch.Tensor, optional
                model's validation input of shape (batch_size, n_features). Default is None.
            - y_val: torch.Tensor, optional
                model's validation output of shape (batch_size, 1). Default is None.
        OUTPUT:
            - None.
        '''
        #Initialize the best model
        self.best_model = self.mdl.state_dict()
        self.best_loss = np.inf
        #Initialize the patience counter
        self.patience = 0
        #Initialize the training and validation losses
        self.train_loss = []
        if isinstance(y_val_, torch.Tensor):
            self.val_loss = []
        else:
            self.val_loss = None
        # Set the wandb environment if required:
        if wandb_save:
            wandb.init(project='clustering_unconditional_crps_cCond_ClassDegree', config=self.params)
            wandb.watch([self.mdl], log='all')
        #Understand if I'm using single or multiple batches
        single_batch = (self.params['batch_size'] == -1) or\
            (self.params['batch_size'] >= y_train_.shape[0])
        # Eventually, create the train indices list
        if not single_batch: indices = torch.arange(y_train_.shape[0])
        # Drawn the points where evaluate the CRPS
        adj_std = torch.std(y_train_) * self.params['crps_std_scale']
        z_train = torch.linspace(
            -adj_std, adj_std, self.params['crps_points'], dtype=torch.float64,
            requires_grad = self.params['pdf_constr'] ).unsqueeze(dim=1).to(self.dev)
        if isinstance(y_val_, torch.Tensor):
            z_val = torch.distributions.uniform.Uniform(-adj_std, adj_std).sample(
                (self.params['crps_points'],)).unsqueeze(dim=1).requires_grad_(
                    requires_grad = self.params['pdf_constr']).to(self.dev)
        # If not mem_save, compute the indicator function for the train and val once at all
        if not mem_save:
            y_train = torch.concat(
                    [torch.where(val <= z_train, 1., 0) for val in y_train_],
                    dim=1).T.to(self.dev).unsqueeze(dim=2).type(torch.float64)
            if isinstance(y_val_, torch.Tensor):
                y_val = torch.concat(
                    [torch.where(val <= z_val, 1., 0).to(self.dev) for val in y_val_],
                    dim=1
                ).T.unsqueeze(dim=2)
            self.loss = self.loss_fast
        else: #Otherwise, use keep the original tensors
            y_train = y_train_
            y_val = y_val_
            self.loss = self.loss_mem_save
        # Set the verbosity if it is provided as a boolean:
        if isinstance(self.verbose, bool):
            self.verbose = int(self.verbose) - 1
        #Train the model
        if self.verbose >= 0: #If verbose is True, then use the progress bar
            from tqdm.auto import tqdm
            it_base = tqdm(range(self.params['n_epochs']), desc='Training the network') #Create the progress bar
        else: #Otherwise, use the standard iterator
            it_base = range(self.params['n_epochs']) #Create the iterator
        for epoch in it_base:
            if (epoch==0) and (self.verbose > 0):
                print_base = '{:<10}{:<15}{:<15}'
                print(print_base.format('Epoch', 'Train Loss', 'Val Loss'))
            # Training
            if single_batch:
                self.train_single_batch(z_train, x_train, y_train)
            else:
                self.train_multi_batch(z_train, y_train, indices)
            # If Validation
            if isinstance(y_val_, torch.Tensor):
                self.mdl.eval() #Set the model in evaluation mode
                if self.params['pdf_constr']:
                    # Compute the validation loss
                    val_loss = 0
                    for n_val, x_value in enumerate(x_val):
                        val_output = self.mdl(
                            torch.cat((z_val,
                                    x_value.unsqueeze(dim=0).repeat(self.params['crps_points'],1)), dim=1)) # Forward pass
                        val_loss += self.loss(val_output, y_val[n_val:n_val+1])
                        pdf = torch.autograd.grad(val_output, z_val,
                                                torch.ones_like(val_output), retain_graph=True)[0]
                        val_loss += torch.mean(nn.ReLU()(-pdf))
                    val_loss /= x_val.shape[0] #Average the loss
                    self.val_loss.append(val_loss.item()) #Save the validation loss
                else:
                    with torch.no_grad():
                        # Compute the validation loss
                        if mem_save:
                            val_loss = self.loss(self.mdl(z_val), y_val, z_val)
                        else:
                            val_loss = 0
                            for n_val, x_value in enumerate(x_val):
                                val_output = self.mdl(
                                    torch.cat((z_val,
                                            x_value.unsqueeze(dim=0).repeat(self.params['crps_points'],1)), dim=1)) # Forward pass
                                val_loss += self.loss(val_output, y_val[n_val:n_val+1])
                            val_loss /= x_val.shape[0] #Average the loss
                        self.val_loss.append(val_loss.item()) #Save the validation loss
                # Update best model and eventually early stopping
                if self.early_stopping(val_loss):
                    if self.verbose >= 0: print(f"Early stopping at epoch {epoch+1}")
                    break
            
            #Eventually, wandb save
            if wandb_save:
                to_wandb = dict()
                to_wandb[f'train_loss'] = self.train_loss[-1]
                to_wandb[f'val_loss'] = self.val_loss[-1]
                wandb.log(to_wandb)
            else: #Otherwise
                # Update best model and eventually early stopping
                if self.early_stopping(self.train_loss[-1]):
                    if self.verbose >= 0: print(f"Early stopping at epoch {epoch+1}")
                    break
            # Eventually, print the losses
            if self.verbose > 0:
                if (epoch+1) % self.verbose == 0:
                    self.print_losses(epoch+1)
        #After the training, load the best model and return its losses
        self.final_ops(z_train, x_train, y_train, z_val, x_val, y_val, mem_save)
        #Eventually, wandb save
        if wandb_save:
            to_wandb = dict()
            to_wandb[f'train_loss'] = self.train_loss[-1]
            to_wandb[f'val_loss'] = self.val_loss[-1]
            wandb.log(to_wandb)
        #Eventually, finish the wandb run
        if wandb_save:
            wandb.finish()

    def print_losses(self, epoch):
        print('{:<10}{:<15}{:<15}'.format(
            epoch, format(self.train_loss[-1], '.20f')[:10],
            format(self.val_loss[-1], '.20f')[:10] if not isinstance(
                self.val_loss, type(None)) else '-'))

    def final_ops(self, z_train, x_train, y_train, z_val, x_val, y_val, mem_save):
        # Recover the best model
        self.mdl.load_state_dict(self.best_model)

        # Compute the train loss of the best model
        z_train.requires_grad_(True)
        if mem_save:
            train_loss = self.loss(output, y_train, z_train)
        else:
            train_loss = 0
            for n_val, x_value in enumerate(x_train):
                output = self.mdl(
                    torch.cat((z_train,
                            x_value.unsqueeze(dim=0).repeat(self.params['crps_points'],1)), dim=1)) # Forward pass
                train_loss += self.loss(output, y_train[n_val:n_val+1])
                pdf = torch.autograd.grad(output, z_train,
                                        torch.ones_like(output), retain_graph=True)[0]
                train_loss += torch.mean(nn.ReLU()(-pdf))
            train_loss /= x_train.shape[0] #Average the loss
        self.train_loss.append(train_loss.item())

        # Compute the val loss of the best model
        z_val.requires_grad_(True)
        if mem_save:
            val_loss = self.loss(val_output, y_val, z_val)
        else:
            val_loss = 0
            for n_val, x_value in enumerate(x_val):
                val_output = self.mdl(
                    torch.cat((z_val,
                            x_value.unsqueeze(dim=0).repeat(self.params['crps_points'],1)), dim=1)) # Forward pass
                val_loss += self.loss(val_output, y_val[n_val:n_val+1])
                pdf = torch.autograd.grad(val_output, z_val,
                                        torch.ones_like(val_output), retain_graph=True)[0]
                val_loss += torch.mean(nn.ReLU()(-pdf))
            val_loss /= x_val.shape[0] #Average the loss
        self.val_loss.append(val_loss.item())

        # Print the loss
        print('\n')
        self.print_losses('Final')
        
    def draw(self, x0, num_points, z_min=-10.0, z_max=10.0, N_grid=20_000, seed=None):
        """
        Generate a path of points from a conditional distribution represented by a neural network mdl.
        
        Args:
            mdl (nn.Module): Neural network representing the conditional CDF.
            x0 (float): Starting point x_0.
            num_points (int): Number of points to generate in the path.
            z_min (float): Minimum value of z to consider in the search.
            z_max (float): Maximum value of z to consider in the search.
            N_grid (int): Number of points to use in the search grid.
            seed (int): Seed for the random number generator.
            
        Returns:
            torch.Tensor: Tensor of generated points.
        """
        if not isinstance(seed, type(None)):
            torch.manual_seed(seed) #Set the seed
            torch.cuda.manual_seed_all(seed)
        
        # Create a grid of z values
        z_values = torch.linspace(z_min, z_max, N_grid, device=self.dev, dtype=torch.float64)
        z_values = z_values.unsqueeze(1)  # Make it (N_grid, 1)
        
        # Initialize the path with the starting point
        path = [x0]
        
        for _ in range(1, num_points):
            # Get the previous point
            x_prev = path[-1]
            
            # Generate a uniform random sample in [0, 1]
            uniform_sample = torch.rand(1, device=self.dev).item()
            
            # Create the input tensor for the model (z_values, x_prev)
            inputs = torch.cat([z_values, torch.full((N_grid, 1), x_prev, device=self.dev)], dim=1)
            
            # Evaluate the CDF for each z in the grid given x_prev
            with torch.no_grad():
                cdf_values = self.mdl(inputs).squeeze(1)  # Make it (N_grid,)
            
            # Function to find the z corresponding to the uniform sample using binary search
            def find_z(cdf_value):
                low, high = 0, N_grid - 1
                while low < high:
                    mid = (low + high) // 2
                    if cdf_values[mid] < cdf_value:
                        low = mid + 1
                    else:
                        high = mid
                return z_values[low].item()
            
            # Find the z corresponding to the uniform sample
            next_point = find_z(uniform_sample)
            
            # Append the generated point to the path
            path.append(next_point)
        
        sampled_points = torch.tensor(path, device=self.dev)
        return sampled_points
    
    def __call__(self, z_test, x_test):
        '''
        Predict the quantile forecast and the expected shortfall.
        INPUT:
            - x_test: torch.Tensor
                input of the model; shape (batch_size, n_features).
        OUTPUT:
            - qf: ndarray
             quantile forecast of the model.
            - ef: ndarray
                expected shortfall predicted by the model.
        '''
        return self.mdl(torch.cat((z_test,
                                    x_test), dim=1))

#%% Conditional- Define the network input and target

del(df)

log_ret = 100 * df_w.LogReturn.dropna().values
log_ret = pd.Series(log_ret, index=df_w.index)

#cond_cl = df_w.Pred.dropna().values
cond_cl = df_w.ClassDegree.dropna().values
cond_cl = pd.Series(cond_cl, index=df_w.index)

y_train = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=42)) &\
            (log_ret.index < end_date-pd.Timedelta(days=14))].values.astype(np.float64)).to(device)
x_train = torch.from_numpy(
    cond_cl[(cond_cl.index >= end_date-pd.Timedelta(days=42)) &\
            (cond_cl.index < end_date-pd.Timedelta(days=14))].values.astype(np.float64)).to(device)
x_train = x_train[ y_train != 0] # Remove the zeros
y_train = y_train[ y_train != 0]
x_train = x_train[:len(x_train)//10]; logging.info('Using a reduced training set')
y_train = y_train[:len(y_train)//10]

y_val = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=14)) &\
            (log_ret.index < end_date-pd.Timedelta(days=7))].values.astype(np.float64)).to(device)
x_val = torch.from_numpy(
    cond_cl[(cond_cl.index >= end_date-pd.Timedelta(days=14)) &\
            (cond_cl.index < end_date-pd.Timedelta(days=7))].values.astype(np.float64)).to(device)
x_val = x_val[ y_val != 0] # Remove the zeros
y_val = y_val[ y_val != 0]
x_val = x_val[:len(x_val)//10]; logging.info('Using a reduced validation set')
y_val = y_val[:len(y_val)//10]

y_test_ = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=7)) &\
            (log_ret.index < end_date)].values.astype(np.float64)).to(device)
x_test_ = torch.from_numpy(
    cond_cl[(cond_cl.index >= end_date-pd.Timedelta(days=7)) &\
            (cond_cl.index < end_date)].values.astype(np.float64)).to(device)
x_test_ = x_test_[ y_test_ != 0 ] # Remove the zeros
y_test_ = y_test_[ y_test_ != 0 ]
z_test = torch.linspace(-torch.std(y_train), torch.std(y_train), 1000, dtype=torch.float64,
                        requires_grad=True ).unsqueeze(dim=1).to(device)
y_test = torch.concat(
    [torch.where(val <= z_test, 1., 0) for val in y_test_], dim=1
    ).T.to(device).unsqueeze(dim=2).type(torch.float64)

x_tv = torch.concatenate([x_train, x_val])
y_tv = torch.concatenate([y_train, y_val])

#%% Conditional - Test the best models

# Define the optimal hyperparameters (actually, I've found params0 = params1)
params_deg = {'activation':'tanh', 'reg_type':'l1_l2', 'batch_size':-1,
            'dropout':0.5, 'reg':[1e-5, 1e-6], 'lr':3e-3,
            'initializer':'glorot_normal', 'layers':[2, 10, 10, 10, 1],
            'optimizer':'adam', 'out_activation':'sigmoid', 'n_epochs':3_000,
            'pdf_constr':False, 'crps_points':100, 'crps_std_scale':3, 'patience':20}

# Create and train the models
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
mdl = NeuralCondCDF(params_deg, device, verbose=20)
mdl.fit(x_train, y_train, x_val, y_val, wandb_save=False)

# Plot CDF and PDF
z_plot = torch.linspace(-torch.std(y_train)*3, torch.std(y_train)*3, 1000, dtype=torch.float64,
                        requires_grad=True ).unsqueeze(dim=1).to(device)
# Compute the CDF
F_z_batch0 = mdl(z_plot)
f_z_batch0 = torch.autograd.grad(F_z_batch0, z_plot0, grad_outputs=torch.ones_like(F_z_batch0))[0]
# sort F_z_batch
F_z_batch0, _ = torch.sort(F_z_batch0)
f_z_batch0, _ = torch.sort(f_z_batch0)
# convert to numpy both z_batch and F_z_batch
F_z_batch0 = F_z_batch0.cpu().detach().numpy()
f_z_batch0 = f_z_batch0.cpu().detach().numpy()
z_batch0 = z_plot0.cpu().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# plot the CDF
sub_p = 0
sns.lineplot(x=z_batch0.flatten(), y=F_z_batch0.flatten(), ax=ax[sub_p], label='Class 0')
ax[sub_p].legend()

# plot the pdf
sub_p = 1
sns.lineplot(x=z_batch0.flatten(), y=f_z_batch0.flatten(), ax=ax[sub_p], label='Class 0')
ax[sub_p].legend()

plt.suptitle('Unconditional Neural CDF and PDF - Clusters Comparison')
#plt.savefig('../figures/neural_cdf_uncond_cluster.png', dpi=150)
plt.show()


