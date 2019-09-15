#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pylab
import matplotlib.pyplot as plt
from numpy import seterr,isneginf,array

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
    
def first_graph(classifieur2, data, classifieur):
    (min_x1,max_x1) = (min(data[:,0]),max(data[:,0]))
    yaxis=np.zeros((data.shape[0],1))
    thegrid = np.linspace(min_x1,max_x1,50)

    
    thegrid = np.vstack(thegrid)
    les_comptes = classifieur2.compute_predictions(thegrid,0.1)
    prob_density=np.exp(les_comptes[:,2])
    les_comptes2 = classifieur2.compute_predictions(thegrid,0.02)
    prob_density2=np.exp(les_comptes2[:,2])
    les_comptes3 = classifieur2.compute_predictions(thegrid,0.5)
    prob_density3=np.exp(les_comptes3[:,2])
    les_comptes4 = classifieur.compute_predictions(thegrid)
    prob_density4=np.exp(les_comptes4[:,2])
    h2= pylab.plot(thegrid[:,0],prob_density2,label='parzen sigma=0.02')
    h1= pylab.plot(thegrid[:,0],prob_density,label='parzen sigma=0.1')
    h3= pylab.plot(thegrid[:,0],prob_density3,label='parzen sigma=0.5')
    h4= pylab.plot(thegrid[:,0],prob_density4,label='gaussian diagonal')
	# Les points d'entrainment
    h5= pylab.scatter(data[:,0], yaxis, marker = "o", s=50,label='data points')
    pylab.xlabel('Valeur')
    pylab.ylabel('Densite')
    pylab.legend()
    pylab.title('Courbes de densite')
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
train_cols = [0]
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
first_graph(classifieur2,iris[100:,train_cols],classifieur)