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
#Cette classe applique la formule de Bayes et est inspirée de ce que nous avons vu en démo
class classif_bayes:

    def __init__(self,modeles_mv, priors):
        self.modeles_mv = modeles_mv
        self.priors = priors
        if len(self.modeles_mv) != len(self.priors):
            print 'Le nombre de modeles MV doit etre egale au nombre de priors!'
        
        self.n_classes = len(self.modeles_mv)

    def compute_predictions(self, test_data, eval_by_group=False):
        log_pred = np.empty((test_data.shape[0],self.n_classes))
        for i in range(self.n_classes):
            log_pred[:,i] = self.modeles_mv[i].compute_predictions(test_data) +  np.log(self.priors[i])
        return log_pred
#Cette classe applique la formule de Bayes et est inspirée de ce que nous avons vu en démo
#Puisque L'application de l'estimation de Parzan est différente nous avons trouvé plus simple d'effectuer les calculs point par point au lieu de le faire par modèle    
class classif_bayes_parzen:

    def __init__(self,modeles_mv, priors):
        self.modeles_mv = modeles_mv
        self.priors = priors
        if len(self.modeles_mv) != len(self.priors):
            print 'Le nombre de modeles MV doit etre egale au nombre de priors!'
        self.n_classes = len(self.modeles_mv)

    def compute_predictions(self, test_data,sigma_sq):
        log_pred = np.empty((test_data.shape[0],self.n_classes))
        for i in range(self.n_classes):
            for j in range(test_data.shape[0]):
                log_pred[j][i] = self.modeles_mv[i].compute_predictions(test_data[j,:],sigma_sq) +  np.log(self.priors[i])
        return log_pred
 
#Cette fonction dessine les régions de décision
#Inspirée fortement des démos
def gridplot(classifieur,train,test,n_points=50):

    train_test = np.vstack((train,test))
    (min_x1,max_x1) = (min(train_test[:,0]),max(train_test[:,0]))
    (min_x2,max_x2) = (min(train_test[:,1]),max(train_test[:,1]))
    xgrid = np.linspace(min_x1,max_x1,num=n_points)
    ygrid = np.linspace(min_x2,max_x2,num=n_points)

	# calcule le produit cartesien entre deux listes
    # et met les resultats dans un array
    thegrid = np.array(combine(xgrid,ygrid))
    les_comptes = classifieur.compute_predictions(thegrid)
    classesPred = np.argmax(les_comptes,axis=1)+1
    pylab.scatter(thegrid[:,0], thegrid[:,1] ,c = classesPred, s=50,  alpha=0.3, edgecolors='none', label="Grille")
	# Les points d'entrainment
    pylab.scatter(train[:,0], train[:,1], c = train[:,-1], marker = 'v', s=50, label= "Train")
    # Les points de test
    pylab.scatter(test[:,0], test[:,1], c = test[:,-1], marker = 's', s=50, label= "Test")
    pylab.title("Region de decision pour Gaussian")
    pylab.legend()
    pylab.show()
#Cette fonction dessine les régions de décision pour l'estimation Parzen 
#Inspirée fortement des démos
#La différence avec la précédente est le paramètre sigma
def gridplot_parzen(classifieur,train,test,sigma, n_points=50):
    
    train_test = np.vstack((train,test))
    (min_x1,max_x1) = (min(train_test[:,0]),max(train_test[:,0]))
    (min_x2,max_x2) = (min(train_test[:,1]),max(train_test[:,1]))
    xgrid = np.linspace(min_x1,max_x1,num=n_points)
    ygrid = np.linspace(min_x2,max_x2,num=n_points)
	# calcule le produit cartesien entre deux listes
    # et met les resultats dans un array
    thegrid = np.array(combine(xgrid,ygrid))
    les_comptes = classifieur.compute_predictions(thegrid,sigma)
    classesPred = np.argmax(les_comptes,axis=1)+1
    pylab.scatter(thegrid[:,0], thegrid[:,1] ,c = classesPred, s=50,  alpha=0.3, edgecolors='none', label="Grille")
	# Les points d'entrainment
    pylab.scatter(train[:,0], train[:,1], c = train[:,-1], marker = 'v', s=50, label= "Train")
    # Les points de test
    pylab.scatter(test[:,0], test[:,1], c = test[:,-1], marker = 's', s=50, label= "Test")
    pylab.legend()
    pylab.title("Region de decision pour Parzen avec hyper-parametre = "+str(sigma))  
    pylab.show()

#Cette fonction dessine la fonction du taux d'erreur par rapport a l'hyper-paramêtre
def sigma_graph(classifieur,train,test,n_points=100):
    (min_x1,max_x1) = (0.0001,3)
    xaxis= np.linspace(min_x1,max_x1,100)
    yaxistest= np.zeros(100)
    yaxistrain= np.zeros(100)
    for i in range(xaxis.shape[0]):
        temp_test = classifieur2.compute_predictions(test[:,:test.shape[1]-1],xaxis[i])
        temp_train= classifieur2.compute_predictions(train[:,:train.shape[1]-1],xaxis[i])
        classesPredtest = np.argmax(temp_test,axis=1)+1
        classesPredtrain = np.argmax(temp_train,axis=1)+1
        yaxistest[i]= ((1-(classesPredtest==test[:,-1]).mean())*100.0)
        yaxistrain[i]= ((1-(classesPredtrain==train[:,-1]).mean())*100.0)
    pylab.plot(xaxis,yaxistest,label='Erreurs Validation')
    pylab.plot(xaxis,yaxistrain,label='Erreurs Entrainement')
	# Les points d'entrainment
    pylab.xlabel('Valeur de sigma')
    pylab.ylabel("Pourcentage d'erreur")
    pylab.legend()
    pylab.title("Taux d'erreur en fonction de l'hyper-parametre pour "+str(test.shape[1]-1)+ " dimensions")
    pylab.show()
    print "Le meilleur taux d'erreur est de "+str(yaxistest.min()) + "% et est atteint a une valeur de sigma de  "+ str(xaxis[yaxistest.argmin()])

#tirée de utilitaires.py de la démo    
def combine(*seqin):
    def rloop(seqin,listout,comb):
        if seqin:                       # any more sequences to process?
            for item in seqin[0]:
                newcomb=comb+[item]     # add next item to current comb
                # call rloop w/ rem seqs, newcomb
                rloop(seqin[1:],listout,newcomb)
        else:                           # processing last sequence
            listout.append(comb)        # comb finished, add to list
    listout=[]                      # listout initialization
    rloop(seqin,listout,[])         # start recursive process
    return listout


# On découpe les données en train/test selon la facon que nous avons vu en démo
iris=np.loadtxt('iris.txt')
np.random.seed(123)
indices1 = np.arange(0,50)
indices2 = np.arange(50,100)
indices3 = np.arange(100,150)

np.random.shuffle(indices1)
np.random.shuffle(indices2)
np.random.shuffle(indices3)
#Division en un ensemble d'entrainement et un ensemble de validation
iris_train1 = iris[indices1[:35]]
iris_test1 = iris[indices1[35:]]
iris_train2 = iris[indices2[:35]]
iris_test2 = iris[indices2[35:]]
iris_train3 = iris[indices3[:35]]
iris_test3 = iris[indices3[35:]]
iris_train = np.concatenate([iris_train1, iris_train2, iris_train3])
iris_test = np.concatenate([iris_test1, iris_test2, iris_test3])
# Premier test: avec deux dimensions
train_cols = [0,2]

#Tout ce qui suit est la création de modèle et l'appel des fonctions crées plus haut
#La création de modèle suit ce que nous avons vu dans les démos
model_classe1=gauss_diag(len(train_cols))
model_classe2=gauss_diag(len(train_cols))
model_classe3=gauss_diag(len(train_cols))
model_classe1.train(iris_train1[:,train_cols])
model_classe2.train(iris_train2[:,train_cols])
model_classe3.train(iris_train3[:,train_cols])

modele_mv=[model_classe1,model_classe2,model_classe3]
priors=[0.3333,0.3333,0.3333]
classifieur=classif_bayes(modele_mv,priors)
log_prob_train=classifieur.compute_predictions(iris_train[:, train_cols])
log_prob_test=classifieur.compute_predictions(iris_test[:, train_cols])
classesPred_train = log_prob_train.argmax(1)+1
classesPred_test = log_prob_test.argmax(1)+1

gridplot(classifieur,iris_train[:, train_cols + [-1]],iris_test[:, train_cols + [-1]],n_points=50)

print "Taux d'erreur gaussien diagonal en 2 dimensions (entrainement) %.2f%%" % ((1-(classesPred_train==iris_train[:,-1]).mean())*100.0)
print "Taux d'erreur gaussien diagonal en 2 dimensions (test) %.2f%%" % ((1-(classesPred_test==iris_test[:,-1]).mean())*100.0)

train_cols = [0,1,2,3]
model_classe1=gauss_diag(len(train_cols))
model_classe2=gauss_diag(len(train_cols))
model_classe3=gauss_diag(len(train_cols))
model_classe1.train(iris_train1[:,train_cols])
model_classe2.train(iris_train2[:,train_cols])
model_classe3.train(iris_train3[:,train_cols])

modele_mv=[model_classe1,model_classe2,model_classe3]
classifieur=classif_bayes(modele_mv,priors)
log_prob_train=classifieur.compute_predictions(iris_train[:, train_cols])
log_prob_test=classifieur.compute_predictions(iris_test[:, train_cols])
classesPred_train = log_prob_train.argmax(1)+1
classesPred_test = log_prob_test.argmax(1)+1

print "Taux d'erreur gaussien diagonal en 4 dimensions(entrainement) %.2f%%" % ((1-(classesPred_train==iris_train[:,-1]).mean())*100.0)
print "Taux d'erreur gaussien diagonal en 4 dimensions(test) %.2f%%" % ((1-(classesPred_test==iris_test[:,-1]).mean())*100.0)
train_cols = [0,2]
model_classe4=gauss_iso_parzen(len(train_cols))
model_classe5=gauss_iso_parzen(len(train_cols))
model_classe6=gauss_iso_parzen(len(train_cols))
model_classe4.train(iris_train1[:,train_cols])
model_classe5.train(iris_train2[:,train_cols])
model_classe6.train(iris_train3[:,train_cols])
modele_mvp=[model_classe4,model_classe5,model_classe6]
classifieur2=classif_bayes_parzen(modele_mvp,priors)

gridplot_parzen(classifieur2,iris_train[:, train_cols + [-1]],iris_test[:, train_cols + [-1]],1,n_points=50)
log_prob_train=classifieur2.compute_predictions(iris_train[:, train_cols],1)
log_prob_test=classifieur2.compute_predictions(iris_test[:, train_cols],1)
classesPred_train = log_prob_train.argmax(1)+1
classesPred_test = log_prob_test.argmax(1)+1

print "Taux d'erreur parzen avec un hyper-parametre de 1 (entrainement) %.2f%%" % ((1-(classesPred_train==iris_train[:,-1]).mean())*100.0)
print "Taux d'erreur parzen avec un hyper-parametre de 1 (test) %.2f%%" % ((1-(classesPred_test==iris_test[:,-1]).mean())*100.0)

gridplot_parzen(classifieur2,iris_train[:, train_cols + [-1]],iris_test[:, train_cols + [-1]],0.2,n_points=50)
log_prob_train=classifieur2.compute_predictions(iris_train[:, train_cols],0.2)
log_prob_test=classifieur2.compute_predictions(iris_test[:, train_cols],0.2)
classesPred_train = log_prob_train.argmax(1)+1
classesPred_test = log_prob_test.argmax(1)+1

print "Taux d'erreur parzen avec un hyper-parametre de 0.2 (entrainement) %.2f%%" % ((1-(classesPred_train==iris_train[:,-1]).mean())*100.0)
print "Taux d'erreur parzen avec un hyper-parametre de 0.2 (test) %.2f%%" % ((1-(classesPred_test==iris_test[:,-1]).mean())*100.0)

gridplot_parzen(classifieur2,iris_train[:, train_cols + [-1]],iris_test[:, train_cols + [-1]],0.01,n_points=50)
log_prob_train=classifieur2.compute_predictions(iris_train[:, train_cols],0.01)
log_prob_test=classifieur2.compute_predictions(iris_test[:, train_cols],0.01)
classesPred_train = log_prob_train.argmax(1)+1
classesPred_test = log_prob_test.argmax(1)+1

print "Taux d'erreur parzen avec un hyper-parametre de 0.01 (entrainement) %.2f%%" % ((1-(classesPred_train==iris_train[:,-1]).mean())*100.0)
print "Taux d'erreur parzen avec un hyper-parametre de 0.01 (test) %.2f%%" % ((1-(classesPred_test==iris_test[:,-1]).mean())*100.0)
sigma_graph(classifieur2,iris_train[:, train_cols + [-1]],iris_test[:, train_cols + [-1]],n_points=100)

train_cols = [0,1,2,3]
model_classe4=gauss_iso_parzen(len(train_cols))
model_classe5=gauss_iso_parzen(len(train_cols))
model_classe6=gauss_iso_parzen(len(train_cols))
model_classe4.train(iris_train1[:,train_cols])
model_classe5.train(iris_train2[:,train_cols])
model_classe6.train(iris_train3[:,train_cols])
modele_mvp=[model_classe4,model_classe5,model_classe6]
classifieur2=classif_bayes_parzen(modele_mvp,priors)
sigma_graph(classifieur2,iris_train[:, train_cols + [-1]],iris_test[:, train_cols + [-1]],n_points=100)
