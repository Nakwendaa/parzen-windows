#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pylab
import matplotlib.pyplot as plt
from numpy import seterr,isneginf,array
import itertools

#Classe pour la gaussienne diagonale 
#Cette classe a été basée sur le travail fait lors des démos
class gauss_diag:
    #initialise les paramètres utiles
     def __init__(self,n_dims):
		self.n_dims = n_dims
		self.mu = np.zeros((1,n_dims))
		self.sigma_sq = np.ones(n_dims)
     #calcule la moyenne et la variance    
     def train(self, train_data):
       self.mu = np.mean(train_data, axis=0)
       self.sigma_sq =  np.sum((train_data - self.mu) ** 2.0, axis = 0) / train_data.shape[0] 
     #retourne un array log de la densité  
     def compute_predictions(self, test_data):
       c = -self.n_dims * np.log(2*np.pi)/2.0 - np.log(np.prod(self.sigma_sq))/2.0
       log_prob = c - np.sum((test_data -  self.mu)**2.0/ (2.0 * self.sigma_sq),axis=1)
       return log_prob

#Classe pour l'estimateur parzen  
#Cette classe a été basée sur le travail fait lors des démos   
class gauss_iso_parzen:
     def __init__(self,n_dims):
         self.n_dims = n_dims
         self.n_train=1
         self.sigma_sq = 1.0
         #pour avoir plus de flexibité on n'a pas besoin d'avoir la dimension exacte de la taille d'entrainement lors de l'initialisation
         self.train_data = np.zeros((n_dims,n_dims))
#garder en mémoire les données d'entrainement et le nombre de données
     def train(self, train_data):
       self.train_data.resize((train_data.shape[0],train_data.shape[1]))
       self.train_data =train_data
       self.n_train=(self.n_dims * train_data.shape[0])
#retourne le log de prob (et non un array comme pour le gaussien diagonal)
     def compute_predictions(self, test_data,sigma_sq):
#la valeur de sigma peut être changé lors de L'appel de compute_prediction pour faciliter les variations de sigma
        #c1 est un entier
       self.sigma_sq=sigma_sq
       c1=1.0/((2*np.pi)**(self.n_dims/2)*self.sigma_sq**self.n_dims)
       #c2 est un array
       c2 =(-1.0/2)*((np.linalg.norm(self.train_data- test_data, axis=1)**2)/self.sigma_sq**2.0)
       prob_parzen = (1.0/self.n_train)*np.sum(c1*np.exp(c2))
       #gère les erreurs si l'hyper-paramètre est trop petit et qu'on obtienne log(0)
       np.seterr(divide='ignore')
       log_prob=np.log(prob_parzen)
       return log_prob
#class qui appelle le calcul de densité pour retourner un tableau de densité
class density_estimator:

    def __init__(self,modeles_mv):
        self.modeles_mv = modeles_mv       
        self.n_classes = len(self.modeles_mv)
#appelle compute_predictions avec une colonne de données (par modele) à la fois
    def compute_predictions(self, test_data, eval_by_group=False):
        log_pred = np.empty((test_data.shape[0],self.n_classes))
        for i in range(self.n_classes):
            log_pred[:,i] = self.modeles_mv[i].compute_predictions(test_data)
        return log_pred
#class qui appelle le calcul de densité pour retourner un tableau de densité    
class density_estimator_parzen:

    def __init__(self,modeles_mv):
        self.modeles_mv = modeles_mv
        self.n_classes = len(self.modeles_mv)
#appelle compute_predictions avec une donnée à la fois
    def compute_predictions(self, test_data,sigma_sq):
        log_pred = np.empty((test_data.shape[0],self.n_classes))
        for i in range(self.n_classes):
            for j in range(test_data.shape[0]):
                log_pred[j][i] = self.modeles_mv[i].compute_predictions(test_data[j,:],sigma_sq)
        return log_pred
#trace le graphe pour la densité gaussienne
def Density2Dgraph(classifieur, data):
    (min_x1,max_x1) = (min(data[:,0]),max(data[:,0]))
    (min_x2,max_x2) = (min(data[:,1]),max(data[:,1]))
    xgrid = np.linspace(min_x1,max_x1,50)
    ygrid = np.linspace(min_x2,max_x2,50)
#similaire a la fonction combine vu dans les travaux pratiques
    thegrid = np.empty((ygrid.shape[0],xgrid.shape[0]))
    for i in range(xgrid.shape[0]):
        toCal =np.empty((ygrid.shape[0],2))
        for j in range(ygrid.shape[0]):
            toCal[j,0]=xgrid[i]
            toCal[j,1]=ygrid[j]        
        thegrid[:,i]=classifieur.compute_predictions(toCal)[:,2]
    prob_density=np.exp(thegrid)
#dénition des fonctions pour les contours
    X = xgrid
    Y = ygrid
    Z=prob_density[np.where(ygrid==Y)][np.where(xgrid==X)]
    pylab.contour(X,Y,Z,20)
    pylab.scatter(data[:,0], data[:,1], marker = "o", s=50,label='Data')
    pylab.legend()
    pylab.xlabel('x')
    pylab.ylabel('y')
    pylab.title("Estimation de densite en 2D par estimateur parametrique Gaussien diagonal")
    pylab.show()
#similaire à la fonction précédente mais pour l'estimateur Parzan
def Density2Dgraph_Parzen(classifieur, data, sigma):

    (min_x1,max_x1) = (min(data[:,0]),max(data[:,0]))
    (min_x2,max_x2) = (min(data[:,1]),max(data[:,1]))
    xgrid = np.linspace(min_x1,max_x1,50)
    ygrid = np.linspace(min_x2,max_x2,50)
    thegrid = np.empty((ygrid.shape[0],xgrid.shape[0]))
    for i in range(xgrid.shape[0]):
        toCal =np.empty((ygrid.shape[0],2))
        for j in range(ygrid.shape[0]):
            toCal[j,0]=xgrid[i]
            toCal[j,1]=ygrid[j]        
        thegrid[:,i]=classifieur.compute_predictions(toCal,sigma)[:,2]
    prob_density=np.exp(thegrid)
    X = xgrid
    Y = ygrid
    Z=prob_density[np.where(ygrid==Y)][np.where(xgrid==X)]
    pylab.contour(X,Y,Z,20)
    pylab.scatter(data[:,0], data[:,1], marker = "o", s=50,label='Data')
    pylab.xlabel('x')
    pylab.ylabel('y')
    pylab.title("Estimation de densite en 2D par estimateur Parzan avec un hyper parametre de "+ str(sigma))
    pylab.legend()
    pylab.axis('equal')
    pylab.show()
  
#le traitement des données a été faite de la façon vue dans les travaux pratiques
iris=np.loadtxt('iris.txt')
indices1 = np.arange(0,50)
indices2 = np.arange(50,100)
indices3 = np.arange(100,150)
iris_train1 = iris[indices1[:]]
iris_train2 = iris[indices2[:]]
iris_train3 = iris[indices3[:]]
iris_train = np.concatenate([iris_train1, iris_train2, iris_train3])
train_cols = [0,2]
model_classe1=gauss_diag(len(train_cols))
model_classe2=gauss_diag(len(train_cols))
model_classe3=gauss_diag(len(train_cols))
model_classe1.train(iris_train1[:,train_cols])
model_classe2.train(iris_train2[:,train_cols])
model_classe3.train(iris_train3[:,train_cols])
modele_mv=[model_classe1,model_classe2,model_classe3]
classifieur=density_estimator(modele_mv)  
model_classe4=gauss_iso_parzen(len(train_cols))
model_classe5=gauss_iso_parzen(len(train_cols))
model_classe6=gauss_iso_parzen(len(train_cols))
model_classe4.train(iris_train1[:,train_cols])
model_classe5.train(iris_train2[:,train_cols])
model_classe6.train(iris_train3[:,train_cols])
modele_mvp=[model_classe4,model_classe5,model_classe6]
classifieur2=density_estimator_parzen(modele_mvp)
#appel des fonctions pour faire les graphes
Density2Dgraph(classifieur,iris[100:,train_cols])
Density2Dgraph_Parzen(classifieur2,iris[100:,train_cols],0.04)
Density2Dgraph_Parzen(classifieur2,iris[100:,train_cols],0.3)
Density2Dgraph_Parzen(classifieur2,iris[100:,train_cols],1)