#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:56:25 2017

@author: Andre Lopes - 45617 Nelson Coquenim - 45694
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage.io import imread
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metr
from sklearn.neighbors import NearestNeighbors



def plot_classes(labels,lon,lat, alpha=0.5, edge = 'k'):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = imread("Mollweide_projection_SW.jpg")        
    plt.figure(figsize=(10,5),frameon=False)    
    x = lon/180*np.pi
    y = lat/180*np.pi
    ax = plt.subplot(111, projection="mollweide")
    print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x,y)).T)
    print(np.min(np.vstack((x,y)).T,axis=0))
    print(np.min(t,axis=0))
    clims = np.array([(-np.pi,0),(np.pi,0),(0,-np.pi/2),(0,np.pi/2)])
    lims = ax.transData.transform(clims)
    plt.close()
    plt.figure(figsize=(10,5),frameon=False)    
    plt.subplot(111)
    plt.imshow(img,zorder=0,extent=[lims[0,0],lims[1,0],lims[2,1],lims[3,1]],aspect=1)        
    x = t[:,0]
    y= t[:,1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0   
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        plt.plot(x[mask], y[mask],'o', markersize=4, mew=1,zorder=1,alpha=alpha, markeredgecolor=edge)
        ix = ix+1                    
    mask = np.logical_not(nots)    
    if np.sum(mask)>0:
        plt.plot(x[mask], y[mask], '.', markersize=1, mew=1,markerfacecolor='w', markeredgecolor=edge)
    plt.axis('off')
    
def transf_earth_coord(Xs):
    RADIUS = 6371
    
    Xs_xyz = np.zeros((Xs.shape[0], Xs.shape[1] + 1))
    Xs_xyz[:,:-1] = Xs
    
    Xs_xyz[:,0] = RADIUS * np.cos(Xs[:,0] * np.pi/180) *  np.cos(Xs[:,1] * np.pi/180)
    Xs_xyz[:,1] = RADIUS * np.cos(Xs[:,0] * np.pi/180) * np.sin(Xs[:,1] * np.pi/180)
    Xs_xyz[:,2] = RADIUS * np.sin(Xs[:,0] * np.pi/180)
    return Xs_xyz
    
def plot_errors(X, errors, title, xlabel):
    plt.figure(1, figsize = (12,8), frameon = False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.gca().yaxis.grid(True)

    plt.plot(X, errors[:,0],"-r", label = "Silhouette")
    plt.plot(X, errors[:,1],"-y", label = "Rand Index")
    plt.plot(X, errors[:,2],"-b", label = "Precision")
    plt.plot(X, errors[:,3],"-g", label = "Recall")
    plt.plot(X, errors[:,4],"-c", label = "F1")
    plt.plot(X, errors[:,5],"-m", label = "Adjusted Rand Index")

    x1,x2,y1,y2 = plt.axis()
    ymin = -0.25 if xlabel == "ε" else 0 #change scale DBSCAN

    plt.axis((x1,x2,ymin,1))
    
    plt.legend(fontsize = 15, loc = 'upper right', ncol=1)

    plt.savefig(title + ".png")
    plt.show()
    plt.close()


data = pd.read_csv("tp2_data.csv", usecols = [2,3,23]).as_matrix()

with_noise = True

if with_noise:
    Xs = data[:,:-1]
    Fs = data[:,-1]
    plot_noise = "_with_noise"
else:
    Xs = data[:,:-1]
    mask = data[:,-1] != -1
    Fs = data[mask, -1]
    Xs = Xs[mask,:]
    plot_noise = "_without_noise"


Xs_xyz = transf_earth_coord(Xs)

#fig = plt.figure(figsize=(6,4))
#ax = fig.add_subplot(111, projection = '3d')
#ax.set_aspect('equal')
#ax.scatter(Xs_xyz[:,0], Xs_xyz[:,1], Xs_xyz[:,2], s=5)
#plt.show()

def score(X, sample_cluster, i):
    sil_score = metr.silhouette_score(Xs_xyz, sample_cluster)
    adj_rand_score = metr.adjusted_rand_score(Fs , sample_cluster)
    TP,FP,FN,TN = confusion_matrix(sample_cluster) #não sei se se pode dar este nome
    N = len(sample_cluster)
    rand_index = (TP + TN) / (N*(N-1)/2)
    precision = TP/(FP + TP)
    recall = TP/(FN + TP)
    F1 = (precision * recall)/(precision + recall)
    errors.append([i, sil_score, rand_index, precision, recall, F1, adj_rand_score])
    
    
def confusion_matrix(sample_cluster):
    TP = FP = FN = TN = 0
    for i in range(len(sample_cluster)):
        for j in range(i + 1, len(sample_cluster)):
            if (sample_cluster[i] == sample_cluster[j] and Fs[i] == Fs[j]):
                TP+=1
            elif(sample_cluster[i] != sample_cluster[j] and Fs[i] != Fs[j]):
                TN+=1
            elif(sample_cluster[i] == sample_cluster[j] and Fs[i] != Fs[j]):
                FP+=1
            elif(sample_cluster[i] != sample_cluster[j] and Fs[i] == Fs[j]):
                FN+=1
                
                
    return TP, FP, FN, TN

##------------K-Means--------------------------
errors = []
for k in np.arange(5, 50, 2):
    kmeans = KMeans(n_clusters=k).fit(Xs_xyz)
    sample_cluster = kmeans.predict(Xs_xyz)
    score(Xs_xyz, sample_cluster, k)

errors_np = np.array(errors)
plot_errors(errors_np[:,0], errors_np[:,1:], "Errors_K-means" + plot_noise, "Number of clusters")

argmax_ARI = np.argmax(errors_np[:,6])
best_ARI = errors_np[argmax_ARI, 6]
best_k = int(errors_np[argmax_ARI, 0])

kmeans = KMeans(n_clusters=best_k).fit(Xs_xyz)
sample_cluster = kmeans.predict(Xs_xyz)
plot_classes(sample_cluster, Xs[:,1], Xs[:,0])
plt.savefig("best_k_" + str(best_k) + "_ARI_" + str(round(best_ARI, 2)) + "_" + plot_noise + ".png", dpi = 300)
plt.close()
    
#------------DBSCAN--------------------------

##----Eps selection---
#min_pts = 4
#nbrs = NearestNeighbors(n_neighbors=min_pts).fit(Xs_xyz)
#distances, indexes = nbrs.kneighbors(Xs_xyz)
#_4_dist = distances[:,-1] #Select distance to the furthest k_point
#ix = _4_dist.argsort()[::-1] #Sort and then invert to be in descending order
#
#plt.figure(1, figsize = (12,8), frameon = False)
#frame = plt.gca()
#plt.title("Sorted 4-Dist Graph", fontsize = 20)
#plt.ylabel("4-Dist", fontsize = 15)
#frame.plot(np.arange(_4_dist.shape[0]), _4_dist[ix], ".r")
#frame.axes.get_xaxis().set_ticks([]) # ignore X values
#
#plt.savefig("sorted_4-dist_graph.png")
#plt.show()


errors = []
for eps in np.arange(100, 400, 20):
    db = DBSCAN(eps=eps).fit(Xs_xyz)
    sample_cluster = db.labels_
    score(Xs_xyz, sample_cluster, eps)
    
errors_np = np.array(errors)
plot_errors(errors_np[:,0], errors_np[:,1:], "Errors_DBSCAN" + plot_noise, "ε")

argmax_ARI = np.argmax(errors_np[:,6])
best_ARI = errors_np[argmax_ARI, 6]
best_eps = int(errors_np[argmax_ARI, 0])

db = DBSCAN(eps=best_eps).fit(Xs_xyz)
sample_cluster = db.labels_
plot_classes(sample_cluster, Xs[:,1], Xs[:,0])
plt.savefig("best_ε_" + str(best_eps) + "_ARI_" + str(round(best_ARI, 2)) + "_" + plot_noise + ".png", dpi = 300)

plt.close()

#------------GMM--------------------------
errors = []
for n_comp in np.arange(5, 50, 2):
    gmm = GaussianMixture(n_components=n_comp).fit(Xs_xyz)
    sample_cluster = gmm.predict(Xs_xyz)
    score(Xs_xyz, sample_cluster, n_comp)

errors_np = np.array(errors)
plot_errors(errors_np[:,0], errors_np[:,1:], "Errors_GMM" + plot_noise, "#Components")

argmax_ARI = np.argmax(errors_np[:,6])
best_ARI = errors_np[argmax_ARI, 6]
best_n = int(errors_np[argmax_ARI, 0])

gmm = GaussianMixture(n_components=best_n).fit(Xs_xyz)
sample_cluster = gmm.predict(Xs_xyz)
plot_classes(sample_cluster, Xs[:,1], Xs[:,0])
plt.savefig("best_n_" + str(best_n) + "_ARI_" + str(round(best_ARI, 2)) + "_" + plot_noise + ".png", dpi = 300)

plt.close()
