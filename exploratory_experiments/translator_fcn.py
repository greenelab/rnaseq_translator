# import the dependencies
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
# general imports
import warnings
import numpy as np
from numpy import random
import pandas as pd
import scanpy as sc
from anndata import AnnData as ad
from tabulate import tabulate
from scipy.stats import spearmanr, pearsonr
from collections import Counter
# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# programming stuff
import time
import os, sys
import pickle
from pathlib import Path

def linear_model_cell_gene(adata, obs_var, obs_cat_X, obs_cat_Y, num_test, num_train):
    #based on https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    #this fcn withholds genes to train and test

    #extracting data from adata object:
    x = adata[adata.obs[obs_var]==obs_cat_X].X
    y = adata[adata.obs[obs_var]==obs_cat_Y].X

    #taking average across all cells of the same celltype, keeping all genes
    y = np.mean(y, axis=0)
    x = np.mean(x, axis=0)
    #y = y[:,np.newaxis]
    x = x[:,np.newaxis]
    #divide into test and training:

    # Split the data into training/testing sets
    #Split the data into training/testing sets
    x_train = x[:-num_train]
    x_test = x[-num_test:]
    y_train = y[:-num_train]
    y_test = y[-num_test:]
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.figure(figsize = [10,10])
    plt.scatter(x_test, y_test, color="green")
    plt.scatter(x_test, y_pred, color="black")
    plt.plot(x_test, y_pred, color="blue", linewidth=3)
    plt.title(f"LinearReg Prediciton SN SC of CellType {obs_cat_Y} with Genes")
    plt.xlabel(f"{obs_cat_X}")
    plt.ylabel(f"{obs_cat_Y}")
    plt.xticks()
    plt.yticks()
    plt.grid(())
    plt.show()

    return(regr, x_train, y_train, x_test, y_test, y_pred)

def linear_model_cellnum(adata, obs_var, obs_cat_X, obs_cat_Y, pc_test):
    #based on https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    #this fcn withholds cells to test and train
    #extracting data from adata object:
    x = adata[adata.obs[obs_var]==obs_cat_X].X
    y = adata[adata.obs[obs_var]==obs_cat_Y].X
    #separating into training and testing, taking av of all cells in each 
    pc_train = 1- pc_test
    idx1 = int(len(x)*pc_test)
    x_test= np.mean(x[0:idx1,:], axis=0) 
    x_test = x_test[:,np.newaxis]
    idx2 = int(len(x)*pc_train)
    x_train = np.mean(x[idx2:-1,:], axis=0)
    x_train = x_train[:,np.newaxis]
    
    idx1 = int(len(y)*pc_test)
    y_test= np.mean(y[0:idx1,:], axis=0) 
    y_test = y_test[:,np.newaxis]
    idx2 = int(len(y)*pc_train)
    y_train = np.mean(y[idx2:-1,:], axis=0)
    y_train = y_train[:,np.newaxis]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(x_train, y_train)
    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.figure(figsize = [10,10])
    plt.scatter(x_test, y_test, color="green")
    plt.scatter(x_test, y_pred, color="black")
    plt.plot(x_test, y_pred, color="blue", linewidth=3)
    plt.title(f"LinearReg Prediciton SN SC of CellType {obs_cat_Y} with Cells")
    plt.xlabel(f"{obs_cat_X}")
    plt.ylabel(f"{obs_cat_Y}")
    plt.xticks()
    plt.yticks()
    plt.grid(())
    plt.show()

    return(regr, x_train, y_train, x_test, y_test, y_pred)

def linear_model_diffcell_gene(adata, obs_var, x, ct1, ct2, pc_test):
    #based on https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html 
    #extracting data from adata object:
    #extracting data from adata object:
    if x == "single_nucleus" or None:
        xtr = adata[adata.obs[obs_var]==f"single_nucleus_{ct1}"].X #SN same celltype as y for training
        xte = adata[adata.obs[obs_var]==f"single_nucleus_{ct2}"].X #SN, another celltype to test
        ytr = adata[adata.obs[obs_var]==f"single_cell_{ct1}"].X #SC  same celltype as y for training
        yte =adata[adata.obs[obs_var]==f"single_cell_{ct2}"].X#SC another celltype as y to compare predicitons
        y = "single_cell"
        x = "single_nucleus"
    else:
        print("Single cell will be used as X training reference")
        ytr = adata[adata.obs[obs_var]==f"single_nucleus_{ct1}"].X #SN same celltype as y for training
        yte = adata[adata.obs[obs_var]==f"single_nucleus_{ct2}"].X #SN, another celltype to test
        xtr = adata[adata.obs[obs_var]==f"single_cell_{ct1}"].X #SC  same celltype as y for training
        xte =adata[adata.obs[obs_var]==f"single_cell_{ct2}"].X#SC another celltype as y to compare predicitons
        x = "single_cell"
        y = "single_nucleus"

    #taking average across all cells of the same celltype, keeping all geness
    ytr = np.mean(ytr, axis=0)#ct1 in SC
    yte = np.mean(yte, axis=0) #ct2 in SC
    xtr = np.mean(xtr, axis=0)  #ct1 in SN
    xte = np.mean(xte, axis=0) #ct2 in SN
    #divide into test and training:

    #Split the data into training/testing sets
    pc_train = 1 - pc_test
        #separating into training and testing, taking av of all cells in each 
    idx1 = int(len(xte)*pc_test)  #num of 20% fo celltypes
    x_test= xte[0:idx1] #taking average of 20% of cells
    x_test = x_test[:,np.newaxis] #creating new axis so we can fit linear model
    #repeating for the others:
    idx2 = int(len(xtr)*pc_train)
    x_train = xtr[idx2:-1]
    x_train = x_train[:,np.newaxis]

    idx1 = int(len(yte)*pc_test)
    y_test= yte[0:idx1]
    y_test = y_test[:,np.newaxis]
    idx2 = int(len(ytr)*pc_train)
    y_train = ytr[idx2:-1]
    y_train = y_train[:,np.newaxis]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.figure(figsize = [10,10])
    plt.scatter(x_test, y_test, color="green")
    plt.scatter(x_test, y_pred, color="black")
    plt.plot(x_test, y_pred, color="blue", linewidth=3)
    plt.title(f"LinearReg Prediciton using {ct1} in {x} Predicting {ct2} in {y} by Genes")
    plt.xlabel(f"{ct1} in {x}")
    plt.ylabel(f"{ct2} in {y}")
    plt.xticks()
    plt.yticks()
    plt.grid(())
    plt.show()

    return(regr, x_train, y_train, x_test, y_test, y_pred)

def linear_model_diffcellnum(adata, obs_var, x, ct1, ct2, pc_test):
    #based on https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    pc_train = 1 - pc_test
    #extracting data from adata object:
    if x == "single_nucleus" or None:
        xtr = adata[adata.obs[obs_var]==f"single_nucleus_{ct1}"].X #SN same celltype as y for training
        xte = adata[adata.obs[obs_var]==f"single_nucleus_{ct2}"].X #SN, another celltype to test
        ytr = adata[adata.obs[obs_var]==f"single_cell_{ct1}"].X #SC  same celltype as y for training
        yte =adata[adata.obs[obs_var]==f"single_cell_{ct2}"].X#SC another celltype as y to compare predicitons
        y = "single_cell"
        x = "single_nucleus"
    else:
        print("Single cell will be used as X training reference")
        ytr = adata[adata.obs[obs_var]==f"single_nucleus_{ct1}"].X #SN same celltype as y for training
        yte = adata[adata.obs[obs_var]==f"single_nucleus_{ct2}"].X #SN, another celltype to test
        xtr = adata[adata.obs[obs_var]==f"single_cell_{ct1}"].X #SC  same celltype as y for training
        xte =adata[adata.obs[obs_var]==f"single_cell_{ct2}"].X#SC another celltype as y to compare predicitons
        x = "single_cell"
        y = "single_nucleus"

    #separating into training and testing, taking av of all cells in each 
    idx1 = int(len(xte)*pc_test)  #num of 20% fo celltypes
    x_test= np.mean(xte[0:idx1,:], axis=0) #taking average of 20% of cells
    x_test = x_test[:,np.newaxis] #creating new axis so we can fit linear model
    #repeating for the others:
    idx2 = int(len(xtr)*pc_train)
    x_train = np.mean(xtr[idx2:-1,:], axis=0)
    x_train = x_train[:,np.newaxis]

    idx1 = int(len(yte)*pc_test)
    y_test= np.mean(yte[0:idx1,:], axis=0) 
    y_test = y_test[:,np.newaxis]
    idx2 = int(len(ytr)*pc_train)
    y_train = np.mean(ytr[idx2:-1,:], axis=0)
    y_train = y_train[:,np.newaxis]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(x_train, y_train)
    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.figure(figsize = [10,10])
    plt.scatter(x_test, y_test, color="green")
    plt.scatter(x_test, y_pred, color="black")
    plt.plot(x_test, y_pred, color="blue", linewidth=3)
    plt.title(f"LinearReg Prediciton of CellType {ct2} with {ct1}")
    plt.xlabel(f"{ct1} in {x}")
    plt.ylabel(f"{ct2} in {y}")
    plt.xticks()
    plt.yticks()
    plt.grid(())
    plt.show()

    return(regr, x_train, y_train, x_test, y_test, y_pred)    

def linear_model_bulked(adata1, adata2, pc_test, by_what, x,y):

    #Split the data into training/testing sets
    pc_train = 1 - pc_test
    if by_what == "genes":
        axx = 1
        idx1 = int(adata1.X.shape[axx]*pc_test) 
        if idx1 < 1:
            idx1 = 1
        x_test= adata1.X[:,0:idx1]
        x_train= adata1.X[:,idx1:-1]
        print(f"xtrain is {x_train.shape}")
        idx2 = int(adata2.X.shape[axx]*pc_test)
        if idx2 < 1:
            idx2 = 1
            by_what = "genes"
        y_test= adata2.X[:,0:idx2]
        y_train= adata2.X[:,idx2:-1]
        print(f"ytrain is {y_train.shape}")
    elif by_what == "cells":
        axx = 0
        idx1 = int(adata1.X.shape[axx]*pc_test) 
        if idx1 < 1:
            print("Can't separate by cells, doing by genes")
            idx1 = 1    
            by_what = "genes"
        x_test= adata1.X[0:idx1,:]
        x_train= adata1.X[idx1:-1,:]
        print(f"xtrain is {x_train.shape}")
        idx2 = int(adata2.X.shape[axx]*pc_test)
        if idx2 < 1:
            print("Can't separate by cells, doing by genes")
            idx2 = 1
            by_what = "genes"
        y_test = adata2.X[0:idx2,:]
        y_train= adata2.X[idx2:-1,:]
        print(f"ytrain is {y_train.shape}")

    #taking average across all cells of the same celltype, keeping all genes or cells
    y_train= np.mean(y_train, axis=0) #SC
    y_test = np.mean(y_test, axis=0) #SC
    x_train = np.mean(x_train, axis=0) #SN
    x_test = np.mean(x_test, axis=0) #SN

    x_test = x_test[:,np.newaxis] #creating new axis so we can fit linear model
    #repeating for the others:
    x_train = x_train[:,np.newaxis]
    y_test = y_test[:,np.newaxis]
    y_train = y_train[:,np.newaxis]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.figure(figsize = [10,10])
    plt.scatter(x_test, y_test, color="green")
    plt.scatter(x_test, y_pred, color="black")
    plt.plot(x_test, y_pred, color="blue", linewidth=3)
    plt.title(f"LinearReg Predition of Bulked {y} using Bulked {x} by {by_what}")
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    plt.xticks()
    plt.yticks()
    plt.grid(())
    plt.show()

    return(regr, x_train, y_train, x_test, y_test, y_pred)

def make_prop_table(adata, obs):
    num_cell_counter = Counter(adata.obs[obs])
    num_cells = list()
    cell_types = list()
    prop_cells = list()
    tot_count = 0
    tot_prop = 0

    for cell in num_cell_counter:
        num_cells.append(num_cell_counter[cell])
        cell_types.append(cell)
        tot_count = tot_count + num_cell_counter[cell]

    for cell in num_cell_counter:
        proportion = num_cell_counter[cell] / tot_count
        prop_cells.append(proportion)
        tot_prop = tot_prop + proportion

    cell_types.append('Total')
    num_cells.append(tot_count)
    prop_cells.append(tot_prop)
    table = {'Cell_Types': cell_types, 
        'Num_Cells': num_cells, 
        'Prop_Cells': prop_cells}
    table = pd.DataFrame(table)
    print(tabulate(table,  headers='keys', tablefmt='fancy_grid', showindex = True))
    return table        

#funtion from https://github.com/greenelab/sc_bulk_ood/blob/main/evaluation_experiments/pbmc/pbmc_experiment_perturbation.ipynb
def mean_sqr_error(single1, single2):
  return np.mean((single1 - single2)**2)    