import csv
import random
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
import pandas as pd

class stacker(object):
    def __init__(self , x_test , x_train , y_train, id_test, base_clf_list=[], blender_clf_list = [], 
                 random_seed = 0,n_folds = 5, eval_metric='roc_auc'):
        """ initializes a stacker object """
        self.__base_clf_list = base_clf_list
        self.__blender_clf_list = blender_clf_list
        self.__random_seed     = random_seed
        self.__n_folds = n_folds
        self.__X_test = x_test
        self.__X = x_train
        self.__y = y_train
        # Number of training data x Number of classifiers
        self.__blend_train = np.zeros((self.__X.shape[0], len(self.__base_clf_list))) 
         # Number of testing data x Number of classifiers
        self.__blend_test = np.zeros((self.__X_test.shape[0], len(self.__base_clf_list)))
        self.__eval_metric=eval_metric
        self.__clf_fold_auc = {}
        self.__blender_fold_auc = {}
        self.__skf = list(StratifiedKFold(self.__y, self.__n_folds,shuffle=True,random_state=self.__random_seed))
        self.__blender_preds = np.zeros((self.__X_test.shape[0], len(self.__blender_clf_list)))
        self.__id_test = id_test

    def train_all_base_classifiers(self):
        """ trains the Blender creates a blended_test and blended_train df"""
        skf = self.__skf
         # For each classifier, we train the number of fold times (=len(skf))
        for j, clf in enumerate(self.__base_clf_list):
            print 'Training classifier [%s] [%s]' % (j,clf[1])
            # Number of testing data x Number of folds , we will take the mean of the predictions later
            blend_test_j = np.zeros((self.__X_test.shape[0], len(skf))) 
            for i, (train_index, cv_index) in enumerate(skf):
                print 'Fold [%s]' % (i)
                # This is the training and validation set
                X_tr = self.__X[train_index]
                Y_tr = self.__y[train_index]
                X_cv = self.__X[cv_index]
                Y_cv = self.__y[cv_index]
                clf[0].fit(X_tr, Y_tr)
                class_preds = clf[0].predict_proba(X_cv)[:,1]
                eval_metric = self.__eval_metric
                auc_score = roc_auc_score(Y_cv, class_preds)
                print "auc_score for fold:",auc_score
                if clf[1] in self.__clf_fold_auc:
                    self.__clf_fold_auc[clf[1]].append(auc_score)
                else:
                    self.__clf_fold_auc[clf[1]] = [auc_score]
                # This output will be the basis for our blended classifier to train against,
                # which is also the output of our classifiers
                self.__blend_train[cv_index, j] = class_preds
                blend_test_j[:, i] = clf[0].predict_proba(self.__X_test)[:,1]
            # Take the mean of the predictions of the cross validation set
            print "cv_score_mean:",np.mean(self.__clf_fold_auc[clf[1]]),"and cv_score_std:",np.std(self.__clf_fold_auc[clf[1]])
            self.__blend_test[:, j] = blend_test_j.mean(1)

    def add_base_classifer(self,clf):
        self.__base_clf_list.append(clf)
        # add a new column to both blended train and test
        self.__blend_train = np.c_[self.__blend_train,np.zeros(self.__X.shape[0])]
        self.__blend_test = np.c_[self.__blend_test,np.zeros(self.__X_test.shape[0])]
        print 'Training classifier [%s] [%s]' % (len(self.__base_clf_list),clf[1])
        # Number of testing data x Number of folds , we will take the mean of the predictions later
        blend_test_j = np.zeros((self.__X_test.shape[0], len(self.__skf))) 
        for i, (train_index, cv_index) in enumerate(self.__skf):
            print 'Fold [%s]' % (i)
            # This is the training and validation set
            X_tr = self.__X[train_index]
            Y_tr = self.__y[train_index]
            X_cv = self.__X[cv_index]
            Y_cv = self.__y[cv_index]
            clf[0].fit(X_tr, Y_tr)
            class_preds = clf[0].predict_proba(X_cv)[:,1]
            eval_metric = self.__eval_metric
            auc_score = roc_auc_score(Y_cv, class_preds)
            print "auc_score for fold:",auc_score
            if clf[1] in self.__clf_fold_auc:
                self.__clf_fold_auc[clf[1]].append(auc_score)
            else:
                self.__clf_fold_auc[clf[1]] = [auc_score]
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            self.__blend_train[cv_index, len(self.__base_clf_list)-1] = class_preds
            blend_test_j[:, i] = clf[0].predict_proba(self.__X_test)[:,1]
        # Take the mean of the predictions of the cross validation set
        print "cv_score_mean:",np.mean(self.__clf_fold_auc[clf[1]]),"and cv_score_std:",np.std(self.__clf_fold_auc[clf[1]])
        self.__blend_test[:, len(self.__base_clf_list)-1] = blend_test_j.mean(1)
    
    def add_blenders(self,blender):
        self.__blender_clf_list.append(blender)

    def remove_blenders(self,blender):
        self.__blender_clf_list.remove(blender)
    
    def find_cv_scores_all_blenders(self):
        for blender,blendername in self.__blender_clf_list:
            print "blender_Name:",blendername,":"
            for i, (train_index, cv_index) in enumerate(self.__skf):
                X_tr = self.__blend_train[train_index]
                Y_tr = self.__y[train_index]
                X_cv = self.__blend_train[cv_index]
                Y_cv = self.__y[cv_index]
                blender.fit(X_tr, Y_tr)
                class_preds = blender.predict_proba(X_cv)[:,1]
                auc_score = roc_auc_score(Y_cv, class_preds)
                print "Fold",i+1,"CV Score:",auc_score
                if blendername in self.__blender_fold_auc:
                    self.__blender_fold_auc[blendername].append(auc_score)
                else:
                    self.__blender_fold_auc[blendername] = [auc_score]
            print "cv_score_mean:",np.mean(self.__blender_fold_auc[blendername]),"and cv_score_std:",np.std(self.__blender_fold_auc[blendername])
        
    def train_all_blenders(self):
        self.__blender_preds = np.zeros((self.__X_test.shape[0], len(self.__blender_clf_list)))
        for i,blender_obj in enumerate(self.__blender_clf_list):
            blender,blender_name = blender_obj
            print "Training Blender #",i+1,"|",blender_name
            blender.fit(self.__blend_train,self.__y)
            blender_probs = blender.predict_proba(self.__blend_test)[:,1]
            self.__blender_preds[:, i] = blender_probs

    def get_weighted_blender_submission(self,submission_name,req_blender_list=[], blender_weights=0):
        """blender_weights should be a np.array"""
        if len(req_blender_list)==0:
            req_blender_list = [x[1] for x in self.__blender_clf_list]
        blender_dataframe = pd.DataFrame(self.__blender_preds,columns= [x[1] for x in self.__blender_clf_list])
        if blender_weights==0:
            #print blender_dataframe
            #print req_blender_list
            blender_df = blender_dataframe[req_blender_list]
            
            new_blender_array = np.sum(np.array(blender_df),axis=1)/len(req_blender_list)
        else:
            blender_weights = np.array(blender_weights)
            blender_df = blender_dataframe[req_blender_list]
            new_blender_array =np.dot(np.array(blender_df),blender_weights)
        submission = pd.DataFrame({"ID":self.__id_test, "TARGET":new_blender_array})
        submission.to_csv(submission_name, index=False)
        return submission

    ### ADD YOU OWN GETTER SETTER FUNCS.
    def get_blend_train(self):
        return self.__blend_train

    def get_blend_test(self):
        return self.__blend_test

    def get_blended_preds_df(self):
        return pd.DataFrame(self.__blender_preds,columns= [x[1] for x in self.__blender_clf_list])

    def __str__(self):
        res = "init:\n" + "n_folds: " + str(self.__n_folds) +" random_seed:"+str(self.__random_seed)+"\n"
        res += "base classifiers:\n"
        for clf_num,clf in enumerate(self.__base_clf_list):
            res += str(clf_num+1)+". "+str(clf) + " \n"
            for i, auc_score in enumerate(self.__clf_fold_auc[clf[1]]):
                res += "fold_"+str(i+1)+"_auc:"+str(auc_score)+"\n"

        res += "Blender classifiers:\n"
        for clf_num,clf in enumerate(self.__blender_clf_list):
            res += str(clf_num+1)+". "+str(clf) + " \n"
            for i, auc_score in enumerate(self.__blender_fold_auc[clf[1]]):
                res += "fold_"+str(i+1)+"_auc:"+str(auc_score)+"\n"
        
        return res
