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
    
    #    plt.ylabel("")
    plt.plot(X, errors[:,0],"-r", label = "Silhouette")
    plt.plot(X, errors[:,1],"-y", label = "Rand Index")
    plt.plot(X, errors[:,2],"-b", label = "Precision")
    plt.plot(X, errors[:,3],"-g", label = "Recall")
    plt.plot(X, errors[:,4],"-c", label = "F1")
    plt.plot(X, errors[:,5],"-m", label = "Adjusted Rand Index")

    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-1,1))
#    plt.axis((x1,x2,0,1))
    
    plt.legend(fontsize = 15, loc = 'lower right', ncol=1)

    plt.savefig(title + ".png")
    plt.show()
    plt.close()


data = pd.read_csv("tp2_data.csv", usecols = [2,3,23]).as_matrix()

Xs = data[:,:-1]
Fs = data[:,-1]

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
    errors.append([i, sil_score,rand_index, precision, recall, F1, adj_rand_score])
    
    
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

#------------K-Means--------------------------

#errors = []
#for k in np.arange( int(96*0.90), int(96*1.05), 10):
#    kmeans = KMeans(n_clusters=k).fit(Xs_xyz)
#   # plot_classes(kmeans.predict(Xs_xyz), Xs[:,1], Xs[:,0])
#    sample_cluster = kmeans.predict(Xs_xyz)
#    
#    score(Xs_xyz, sample_cluster, k)
#
#errors_np = np.array(errors)
#plot_errors(errors_np[:,0], errors_np[:,1:], "Errors_K-means", "Number of clusters")
    
#------------DBSCAN--------------------------

errors = []
for eps in np.arange(0.3, 4, 0.3):
    db = DBSCAN(eps=eps).fit(Xs_xyz)
   # plot_classes(kmeans.predict(Xs_xyz), Xs[:,1], Xs[:,0])
    sample_cluster = db.labels_
    score(Xs_xyz, sample_cluster, eps)
    
errors_np = np.array(errors)
plot_errors(errors_np[:,0], errors_np[:,1:], "Errors_DBSCAN", "ε")

#------------GMM--------------------------
#errors = []
#for n_comp in np.arange(2, 100, 20):
#    gmm = GaussianMixture(n_components=n_comp).fit(Xs_xyz)
#   # plot_classes(kmeans.predict(Xs_xyz), Xs[:,1], Xs[:,0])
#    sample_cluster = gmm.predict(Xs_xyz)
#    score(Xs_xyz, sample_cluster, n_comp)
#
#errors_np = np.array(errors)
#plot_errors(errors_np[:,0], errors_np[:,1:], "Errors_GMM", "#Components")



