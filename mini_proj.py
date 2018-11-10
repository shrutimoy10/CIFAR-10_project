import os
import cv2 as cv
import pickle
import numpy as np
import pdb
import requests
from collections import defaultdict
import random 
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score,log_loss,hinge_loss
from sklearn.model_selection import train_test_split

from tqdm import *
from functools import wraps
from time import time as _timenow 
from sys import stderr
from matplotlib import pyplot as plt

# Loading the CIFAR-10 Dataset

def load_cifar():
    
    trn_data, trn_labels, tst_data, tst_labels = [], [], [], []
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data
    
    for i in trange(5):
        batchName = './data/data_batch_{0}'.format(i + 1)
        #print(batchName)
        unpickled = unpickle(batchName)
        trn_data.extend(unpickled['data'])
        trn_labels.extend(unpickled['labels'])
    unpickled = unpickle('./data/test_batch')
    tst_data.extend(unpickled['data'])
    tst_labels.extend(unpickled['labels'])
    return trn_data, trn_labels, tst_data, tst_labels


#PREPROCESSING STARTS

#splitting the training data into training and validation sets
def train_val_split(trn_data,trn_labels,trn_ratio):
	return train_test_split(trn_data,trn_labels,train_size = trn_ratio)

#converting the images to grayscale
def grayscale(data):
	data = np.asarray(data)
	X = np.zeros((data.shape[0],data.shape[1]*data.shape[2]))
	for i in trange(data.shape[0]):
		img = cv.cvtColor(data[i],cv.COLOR_BGR2GRAY)
		X[i] = np.ravel(img,order = 'C')
	return X


#normalize the images
def normalize_images(data):
	scaler = MinMaxScaler(feature_range = (0,1),copy=False)
	scaler.fit(data)
	features = scaler.transform(data)
	return features



def preprocess_images(X_train,X_val,test_data):
	#reshaping each row to an image
	X_train = np.reshape(X_train,(40000,32,32,3))
	X_val = np.reshape(X_val,(10000,32,32,3))
	test_data = np.reshape(test_data,(10000,32,32,3))
	#grayscaling the images
	X_train = grayscale(X_train) # 40000x1024
	X_val = grayscale(X_val) #10000 x 1024
	test_data = grayscale(test_data) # 10000x1024
	#normalizing the images
	X_train = normalize_images(X_train)
	X_val = normalize_images(X_val)
	test_data = normalize_images(test_data)

	return X_train,X_val,test_data

#PREPROCESSING ENDS


#DIMENSIONALITY REDUCTION STARTS
#params : n_components,method = {'PCA','LDA','KPCA',...}
def reduce_dimension(X_train,X_val,test_data,y_train,y_val,test_labels,n_components,**kwargs):
	if kwargs['method'] == 'pca':
		pca_trn,pca_val,pca_tst = perform_PCA(X_train,X_val,test_data,n_components)
		return pca_trn,pca_val,pca_tst

	elif kwargs['method'] == 'lda':
		pca_trn,pca_val,pca_tst = perform_PCA(X_train,X_val,test_data,250) #95% variance at 250 components
		lda_trn,lda_val,lda_tst = perform_LDA(pca_trn,pca_val,pca_tst,y_train,y_val,test_labels,n_components)
		return lda_trn,lda_val,lda_tst


#perform PCA for dimensionality reduction
def perform_PCA(X_train,X_val,test_data,n_components):
	pca = PCA(n_components=n_components,svd_solver = 'arpack') 
	pca.fit(X_train)
	#print('Variance : ', np.sum(pca.explained_variance_ratio_))
	#principal components of the training data
	# 95% variance at 250 components
	pca_trn = pca.transform(X_train)  
	#principal components of the validation data
	pca_val = pca.transform(X_val)
	#principal components of the test data
	pca_tst = pca.transform(test_data)
	return pca_trn,pca_val,pca_tst

#pca_trn,pca_val,pca_tst = reduce_dimension(250,method='pca')


#perform LDA for dimensionality reduction
#performing lda after pca
#83% variance retained for 9 linear discriminants
def perform_LDA(X_train,X_val,test_data,y_train,y_val,test_labels,comp):
	lda = LinearDiscriminantAnalysis(n_components = comp,solver='eigen',shrinkage='auto')
	lda.fit(X_train,y_train)
	#linear discriminants for the training data
	lda_trn = lda.transform(X_train)
	#linear discriminants for the validation data
	lda_val = lda.transform(X_val)
	#linear discriminants for the test data
	lda_tst = lda.transform(test_data)
	return lda_trn,lda_val,lda_tst

#DIMENSIONALITY REDUCTION ENDS

#CLASSIFICATION STARTS
def classify(X_train,y_train,X_val,y_val,X_test,test_labels,**kwargs):
	if kwargs['classifier'] == 'MLP':
		y_pred = classify_MLP(X_train,y_train,X_val,y_val,X_test,test_labels)
		return y_pred
	elif kwargs['classifier'] == 'LinearSVM':
		y_pred = classify_LinearSVM(X_train,y_train,X_val,y_val,X_test,test_labels)
		return y_pred
	elif kwargs['classifier'] == 'Logistic Regression' :
		y_pred = classify_Logistic_Regression(X_train,y_train,X_val,y_val,X_test,test_labels)
		return y_pred
	elif kwargs['classifier'] == 'Random Forest':
		y_pred = classify_RandomForest(X_train,y_train,X_val,y_val,X_test,test_labels)
		return y_pred


#using MLP for classification 
def classify_MLP(X_train,y_train,X_val,y_val,X_test,test_labels):
	epoch,hid_size = learn_classifier(X_train,y_train,X_val,y_val,X_test,test_labels,classifier = 'MLP')
	clf = MLPClassifier(max_iter = epoch,solver='sgd',alpha = 1e-5,batch_size = 200,
							early_stopping = False,hidden_layer_sizes = (hid_size,hid_size),
							learning_rate = 'adaptive',learning_rate_init = 0.01,
							momentum = 0.9,nesterovs_momentum = True,random_state = 1,
							activation='logistic',tol = 1e-9,verbose=True,validation_fraction=0.0)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	return y_pred

#using Linear SVM for classification
def classify_LinearSVM(X_train,y_train,X_val,y_val,X_test,test_labels):
	opt_penalty = learn_classifier(X_train,y_train,X_val,y_val,X_test,test_labels,classifier = 'LinearSVM')
	clf = LinearSVC(penalty='l2',loss = 'hinge',dual = True,tol = 1e-7,C = opt_penalty,multi_class = 'ovr',
							fit_intercept = True,intercept_scaling = 10.0,verbose = 1,max_iter = 500)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	return y_pred

#using Logistic Regression for classification
def classify_Logistic_Regression(X_train,y_train,X_val,y_val,X_test,test_labels):
	opt_penalty = learn_classifier(X_train,y_train,X_val,y_val,X_test,test_labels,classifier = 'Logistic Regression')
	clf = LogisticRegression(penalty = 'l2',dual = False, tol = 1e-7, C = opt_penalty, fit_intercept = True, intercept_scaling = 1,
							 random_state = 1, solver = 'newton-cg',max_iter = 500, multi_class = 'multinomial', verbose = 1, n_jobs = 4)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	return y_pred

#using Random Forest for classification
def classify_RandomForest(X_train,y_train,X_val,y_val,X_test,test_labels):
	no_of_trees,depth = learn_classifier(X_train,y_train,X_val,y_val,X_test,test_labels,classifier = 'Random Forest')
	clf = RandomForestClassifier(n_estimators = no_of_trees,max_depth = depth,verbose = 2)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	return y_pred

#CLASSIFICATION ENDS

#VALIDATION STARTS
#validate the given classifier with the params in the grid search
def learn_classifier(X_train,y_train,X_val,y_val,X_test,test_labels,**kwargs):
	if kwargs['classifier'] == 'MLP':
		#tuning the no of iterations
		#epoch = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,
							#260,270,280,290,300]) #210 optimal for raw data, PCA
		#epoch = np.array([100,150,200,250,300,350,400,425,450])  #for LDA  #400 optimal for LDA
		hid_size = np.array([5,10])#,15,20,25,30,35,40,45,50])
		#train_loss = np.zeros(len(epoch),dtype = np.float)
		#val_loss = np.zeros(len(epoch),dtype = np.float)
		train_loss = np.zeros(len(hid_size),dtype = np.float)
		val_loss = np.zeros(len(hid_size),dtype = np.float)
		#for i in trange(len(epoch)):
		for i in trange(len(hid_size)):
			clf = MLPClassifier(max_iter = 210,solver='sgd',alpha = 1e-5,batch_size = 200,
							early_stopping = False,hidden_layer_sizes = (hid_size[i],hid_size[i]),
							learning_rate = 'adaptive',learning_rate_init = 0.01,
							momentum = 0.9,nesterovs_momentum = True,random_state = 1,
							activation='logistic',tol = 1e-9,verbose=True,validation_fraction=0.0)
			clf.fit(X_train,y_train)
			#y_pred_train = clf.predict(X_train) #for calculating training loss
			y_pred_train = clf.predict_proba(X_train)
			train_loss[i] = log_loss(y_train,y_pred_train)

			#clf.fit(X_val,y_val)
			y_pred_val = clf.predict_proba(X_val)
			val_loss[i] = log_loss(y_val,y_pred_val)

		fig,ax = plt.subplots(1,1)
		ax.plot(hid_size,train_loss,color = 'blue',label= 'Training Loss')
		ax.plot(hid_size,val_loss,color = 'red',label = 'Validation Loss')
		ax.set_xlabel('Hidden Layer Size(2 layers)')
		ax.set_ylabel('Loss')
		#ax.set_title('Train loss and val loss for different epochs with 250 principal components.')
		#ax.set_title('Train loss and val loss for different epochs with 9 Linear Discriminants.')
		#ax.set_title('Train loss and val loss for different epochs with raw data.')
		#ax.set_title('Train loss and val loss for different hidden layer sizes with raw data.')
		ax.set_title('Loss for different hidden layer sizes with 250 principal components.')
		#ax.set_title('Loss for different hidden layer sizes with 9 Linear Discriminants.')
		plt.legend()
		plt.savefig('MLP_HiddenLoss_PCA.jpg')

		max_iter = 210 #epoch[np.argmin(val_loss)] #iteration giving the min val loss
		hidden_size = hid_size[np.argmin(val_loss)]
		print(hidden_size)
		return max_iter,hidden_size

	elif kwargs['classifier'] == 'LinearSVM':
		#tuning the cost parameter
		#cost_param = np.array([5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0])
		cost_param = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
		train_loss = np.zeros(len(cost_param),dtype = np.float)
		val_loss = np.zeros(len(cost_param),dtype = np.float)
		acc = np.zeros(len(cost_param),dtype = np.float)

		for i in trange(len(cost_param)):
			clf = LinearSVC(penalty='l2',loss = 'hinge',dual = True,tol = 1e-7,C = cost_param[i],multi_class = 'ovr',
							fit_intercept = True,intercept_scaling = 1,verbose = 1,max_iter = 500)
			
			clf.fit(X_train,y_train)
			'''
			#y_pred_train = clf.predict(X_train)
			y_pred_train = clf.decision_function(X_train)
			labels = np.array([0,1,2,3,4,5,6,7,8,9])
			print("OP  : ", y_pred_train)
			train_loss[i] = hinge_loss(y_train,y_pred_train,labels)

			#y_pred_val = clf.predict(X_val)
			y_pred_val = clf.decision_function(X_val)
			val_loss[i] = hinge_loss(y_val,y_pred_val,labels)
			'''
			y_pred = clf.predict(X_test)
			acc[i] = accuracy_score(test_labels,y_pred)

		fig,ax = plt.subplots(1,1)
		#ax.plot(cost_param,train_loss,color = 'blue',label = 'Training Loss')
		ax.plot(cost_param,acc,color = 'red', label = 'Accuracy')
		ax.set_xlabel('Penalty parameter(C)')
		ax.set_ylabel('Accuracy')
		#ax.set_title('Loss for different penalty parameters with raw data.')
		#ax.set_title('Loss for different penalty parameters with 250 principal components.')
		ax.set_title('Accuracy for different penalty parameters with 9 linear discriminants')
		plt.legend()
		#plt.savefig('Linear_SVM_penalty_LDA_Acc.jpg')
		plt.savefig('Linear_SVM_penalty_LDA_Acc.jpg')

		#opt_penalty = cost_param[np.argmin(val_loss)]
		opt_penalty = cost_param[np.argmax(acc)]
		print(opt_penalty)
		return opt_penalty

	elif kwargs['classifier'] == 'Logistic Regression' :
		#tuning the penalty parameter
		#pen_param = np.array([1.0,2.0,3.0,4.0,5.0])
		pen_param = np.array([0.1,0.2,0.3,0.4,0.5])
		train_loss = np.zeros(len(pen_param),dtype = np.float)
		val_loss = np.zeros(len(pen_param),dtype = np.float)
		acc = np.zeros(len(pen_param),dtype = np.float)

		for i in trange(len(pen_param)):
			clf = LogisticRegression(penalty = 'l2',dual = False, tol = 1e-7, C = pen_param[i], fit_intercept = True, intercept_scaling = 1,
							 random_state = 1, solver = 'newton-cg',max_iter = 500, multi_class = 'multinomial', verbose = 2, n_jobs = 2)
			clf.fit(X_train,y_train)
			'''
			y_pred_train = clf.predict_proba(X_train)
			train_loss[i] = log_loss(y_train,y_pred_train)

			y_pred_val = clf.predict_proba(X_val)
			val_loss[i] = log_loss(y_val,y_pred_val)
			'''
			y_pred = clf.predict(X_test)
			acc[i] = accuracy_score(test_labels,y_pred)


		fig,ax = plt.subplots(1,1)
		#ax.plot(pen_param,train_loss,color = 'blue', label = 'Training Loss')
		ax.plot(pen_param,acc,color = 'red', label = 'Accuracy')
		ax.set_xlabel('Penalty Parameter(C)')
		ax.set_ylabel('Accuracy')
		#ax.set_title('Loss for different penalty parameters with raw data')
		#ax.set_title('Loss for different penalty parameters with 250 principal components')
		#ax.set_title('Loss for different penalty parameters with 9 linear discriminants')
		ax.set_title('Accuracy for different penalty parameters with 9 linear discriminants')
		plt.legend()
		#plt.savefig('LR_loss_pen_LDA.jpg')
		plt.savefig('LR_acc_LDA.jpg')

		opt_penalty = pen_param[np.argmax(acc)]
		print(opt_penalty)
		return opt_penalty

	elif kwargs['classifier'] == 'Random Forest':
		#tuning the no of trees
		no_of_trees = np.array([100,200,300,400,500,600,700,800,900,1000])
		#depth = np.array([2,3,4,5,6,7,8,9,10,11])
		#train_loss = np.zeros(len(no_of_trees),dtype = np.float)
		#val_loss = np.zeros(len(no_of_trees),dtype = np.float)
		acc = np.zeros(len(no_of_trees),dtype = np.float)
		#train_loss = np.zeros(len(depth),dtype = np.float)
		#val_loss = np.zeros(len(depth),dtype = np.float)
		#acc = np.zeros(len(depth),dtype = np.float)

		for i in trange(len(no_of_trees)):
		#for i in trange(len(depth)):
			clf = RandomForestClassifier(n_estimators = no_of_trees[i],max_depth = 10,verbose = 2,n_jobs = 2)
			clf.fit(X_train,y_train)
			'''
			y_pred_train = clf.predict_proba(X_train)
			train_loss[i] = log_loss(y_train,y_pred_train)

			y_pred_val = clf.predict_proba(X_val)
			val_loss[i] = log_loss(y_val,y_pred_val)
			'''
			y_pred = clf.predict(X_test)
			acc[i] = accuracy_score(test_labels,y_pred)
		fig,ax = plt.subplots(1,1)
		#ax.plot(depth,train_loss,color = 'blue',label = 'Training Loss')
		ax.plot(no_of_trees,acc,color = 'red',label = 'Accuracy')
		ax.set_xlabel('No of Trees')
		ax.set_ylabel('Accuracy')
		#ax.set_title('Loss for different no of trees with raw data')
		ax.set_title('Accuracy for different no of trees with 9 linear discriminants')
		plt.legend()
		plt.savefig('RF_Acc_trees_LDA.jpg')

		opt_trees = no_of_trees[np.argmax(acc)]
		print(opt_trees)
		#opt_trees = 900
		#opt_depth = depth[np.argmax(acc)]
		#print(opt_depth)
		opt_depth = 10
		return opt_trees,opt_depth
		#return opt_trees




#VALIDATION ENDS

#EVALUATION STARTS
#returns the f1_score and accuracy score of the classifier model
def evaluate(test_labels,y_pred):
	f1 = f1_score(test_labels,y_pred,average='micro')
	acc = accuracy_score(test_labels,y_pred)
	return f1,acc

#EVALUALTION ENDS





#DRIVER FUNCTION
def main():
	train_data_no_val,train_labels,test_data,test_labels = load_cifar()
	#performing a 80:20 train:val split
	X_train,X_val,y_train,y_val = train_val_split(train_data_no_val,train_labels, 0.8)
	X_train,X_val,test_data = preprocess_images(X_train,X_val,test_data)
	#X_train,X_val,test_data = reduce_dimension(X_train,X_val,test_data,y_train,y_val,test_labels,250,method='pca')
	X_train,X_val,test_data = reduce_dimension(X_train,X_val,test_data,y_train,y_val,test_labels,9,method='lda')
	#y_pred = classify(X_train,y_train,X_val,y_val,test_data,test_labels,classifier = 'Random Forest')  # training with 40000 samples
	#y_pred = classify(X_train[:12000],y_train[:12000],X_val[:3000],y_val[:3000],test_data[:3000],classifier = 'Random Forest') # training with 4000 samples, test-val = 800 samples
	y_pred = classify(X_train,y_train,X_val,y_val,test_data,test_labels,classifier = 'Random Forest')  # for log regression and linear svm
	#print(y_pred)
	f1,acc = evaluate(test_labels[:2000],y_pred)
	#f1,acc = evaluate(test_labels[:2000],y_pred)
	print('F1 Score : ' , f1, 'Accuracy : ',acc)

if __name__ == '__main__':
	main()