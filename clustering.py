#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:56:25 2017

@author: Andre Lopes - 45617 Nelson Coquenim - 45694
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skimage.io import imread
import pandas as pd

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
    
    Xs_xyz[:,0] = RADIUS * np.multiply(np.cos(Xs[:,0] * np.pi/180), np.cos(Xs[:,1] * np.pi/180))
    Xs_xyz[:,1] = RADIUS * np.multiply(np.cos(Xs[:,0] * np.pi/180), np.sin(Xs[:,1] * np.pi/180))
    Xs_xyz[:,2] = RADIUS * np.sin(Xs[:,0] * np.pi/180)
    return Xs_xyz

data = pd.read_csv("tp2_data.csv", na_values='?' , usecols = [2,3,23]).as_matrix()


Xs = data[:,:-1]
Fs = data[:,-1]

Xs = transf_earth_coord(Xs)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(Xs[:,0], Xs[:,1], Xs[:,2], s=5)
plt.show()

        

























