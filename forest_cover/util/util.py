import logging
from forest_cover.exception import forest_cover_exception
import yaml
import os,sys
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import scipy.cluster.hierarchy as sch
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as  np
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from forest_cover.constant import LOGISTICS_PARAMS_TUNING,CURRENT_TIME_STAMP,SVC_PARAMS_TUNING,DECISION_TREE_TUNING,RANDOM_FOREST_TUNING,GD_TUNING


def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise forest_cover_exception(e,sys) from e

def grid_cv(algo,params_g,X,y):
    gr=GridSearchCV(estimator=algo,param_grid=params_g)
    gr.fit(X,y)
    return gr.best_params_


def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)

def do_train_0(dataframe):
    try:
        X=dataframe.drop("Cover_Type",axis=1)
        y=dataframe["Cover_Type"]

        # dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
        # plt.title('Dendogram')
        # plt.xlabel('Forests')
        # plt.ylabel('Eucledian distance')
        # plt.savefig("dendogram.png")

        hr = AgglomerativeClustering(n_clusters=2 , affinity='euclidean' ,linkage="ward")
        cluster_df=pd.DataFrame(hr.fit_predict(X),columns=["cluster"])
        K = range(2,10)
        for n_clusters in list(K):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters , affinity='euclidean' ,linkage="ward")
            preds = clusterer.fit_predict(X)
            score = silhouette_score(X, preds)
            with open("silhoute_score.txt","a+") as f:
                f.write("For n_clusters = {}, silhouette score is {}".format(n_clusters, score)+'\n')
            # print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))
        df2=pd.concat([X,y,cluster_df],axis=1)
        cluster_0=df2[df2["cluster"]==0]
        cluster_0.drop("cluster",axis=1,inplace=True)
        # cluster_1=df2[df2["cluster"]==1]
        # cluster_1.drop("cluster",axis=1,inplace=True)
        ##cluster 0
        sm=SMOTETomek(random_state=30)
        X_cluster_0=cluster_0.drop("Cover_Type",axis=1)
        y_cluster_0=cluster_0["Cover_Type"]
        X_cluster_0,y_cluster_0=sm.fit_resample(X_cluster_0,y_cluster_0)
        # X_cluster_1=cluster_1.drop("Cover_Type",axis=1)
        # y_cluster_1=cluster_1["Cover_Type"]

        # X_cluster_1,y_cluster_1=sm.fit_resample(X_cluster_1,y_cluster_1)
        return X_cluster_0,y_cluster_0
    except Exception as e:
        raise forest_cover_exception(e,sys) from e


def do_train_0_train(dataframe):
    try:
        X=dataframe.drop("Cover_Type",axis=1)
        y=dataframe["Cover_Type"]

        # dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
        # plt.title('Dendogram')
        # plt.xlabel('Forests')
        # plt.ylabel('Eucledian distance')
        # plt.savefig("dendogram.png")

        hr = AgglomerativeClustering(n_clusters=2 , affinity='euclidean' ,linkage="ward")
        cluster_df=pd.DataFrame(hr.fit_predict(X),columns=["cluster"])
        K = range(2,10)
        for n_clusters in list(K):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters , affinity='euclidean' ,linkage="ward")
            preds = clusterer.fit_predict(X)
            score = silhouette_score(X, preds)
            with open("silhoute_score.txt","a+") as f:
                f.write("For n_clusters = {}, silhouette score is {}".format(n_clusters, score)+'\n')
            # print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))
        df2=pd.concat([X,y,cluster_df],axis=1)
        # cluster_0=df2[df2["cluster"]==0]
        # cluster_0.drop("cluster",axis=1,inplace=True)
        cluster_1=df2[df2["cluster"]==1]
        cluster_1.drop("cluster",axis=1,inplace=True)
        ##cluster 0
        rm=RandomOverSampler(random_state=30)
        # X_cluster_0=cluster_0.drop("Cover_Type",axis=1)
        # y_cluster_0=cluster_0["Cover_Type"]
        # X_cluster_0,y_cluster_0=rm.fit_resample(X_cluster_0,y_cluster_0)
        X_cluster_1=cluster_1.drop("Cover_Type",axis=1)
        y_cluster_1=cluster_1["Cover_Type"]

        X_cluster_1,y_cluster_1=rm.fit_resample(X_cluster_1,y_cluster_1)
        return X_cluster_1,y_cluster_1
    except Exception as e:
        raise forest_cover_exception(e,sys) from e






def do_train_0_test(dataframe):
    try:
        X=dataframe.drop("Cover_Type",axis=1)
        y=dataframe["Cover_Type"]

        # dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
        # plt.title('Dendogram')
        # plt.xlabel('Forests')
        # plt.ylabel('Eucledian distance')
        # plt.savefig("dendogram.png")

        hr = AgglomerativeClustering(n_clusters=2 , affinity='euclidean' ,linkage="ward")
        cluster_df=pd.DataFrame(hr.fit_predict(X),columns=["cluster"])
        K = range(2,10)
        for n_clusters in list(K):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters , affinity='euclidean' ,linkage="ward")
            preds = clusterer.fit_predict(X)
            score = silhouette_score(X, preds)
            with open("silhoute_score.txt","a+") as f:
                f.write("For n_clusters = {}, silhouette score is {}".format(n_clusters, score)+'\n')
            # print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))
        df2=pd.concat([X,y,cluster_df],axis=1)
        cluster_0=df2[df2["cluster"]==0]
        cluster_0.drop("cluster",axis=1,inplace=True)
        ##cluster 0
        sm=SMOTETomek(random_state=30)
        X_cluster_0=cluster_0.drop("Cover_Type",axis=1)
        y_cluster_0=cluster_0["Cover_Type"]
        vf=y_cluster_0.value_counts()
        X_cluster_0,y_cluster_0=sm.fit_resample(X_cluster_0,y_cluster_0)

        return X_cluster_0,y_cluster_0
    except Exception as e:
        raise forest_cover_exception(e,sys) from e



def do_train_1_train(dataframe):
    try:
        X=dataframe.drop("Cover_Type",axis=1)
        y=dataframe["Cover_Type"]

        # dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
        # plt.title('Dendogram')
        # plt.xlabel('Forests')
        # plt.ylabel('Eucledian distance')
        # plt.savefig("dendogram_1.png")

        hr = AgglomerativeClustering(n_clusters=2 , affinity='euclidean' ,linkage="ward")
        cluster_df=pd.DataFrame(hr.fit_predict(X),columns=["cluster"])
        K = range(2,10)
        for n_clusters in list(K):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters , affinity='euclidean' ,linkage="ward")
            preds = clusterer.fit_predict(X)
            score = silhouette_score(X, preds)
            with open("silhoute_score.txt","a+") as f:
                f.write("For n_clusters = {}, silhouette score is {}".format(n_clusters, score)+'\n')
            # print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))
        df2=pd.concat([X,y,cluster_df],axis=1)
        cluster_1=df2[df2["cluster"]==1]
        cluster_1.drop("cluster",axis=1,inplace=True)
        ##cluster 1
        sm=SMOTETomek(random_state=30)
        X_cluster_1=cluster_1.drop("Cover_Type",axis=1)
        y_cluster_1=cluster_1["Cover_Type"]
        X_cluster_1,y_cluster_1=sm.fit_resample(X_cluster_1,y_cluster_1)
        return X_cluster_1,y_cluster_1
    except Exception as e:
        raise forest_cover_exception(e,sys) from e


    

def do_train_1_test(dataframe):
    try:
        # dataframe=dataframe[dataframe["Cover_Type"]!=3]
        # dataframe=dataframe[dataframe["Cover_Type"]!=4]
        X=dataframe.drop("Cover_Type",axis=1)
        y=dataframe["Cover_Type"]

        # dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
        # plt.title('Dendogram')
        # plt.xlabel('Forests')
        # plt.ylabel('Eucledian distance')
        # plt.savefig("dendogram_1.png")

        hr = AgglomerativeClustering(n_clusters=2 , affinity='euclidean' ,linkage="ward")
        cluster_df=pd.DataFrame(hr.fit_predict(X),columns=["cluster"])
        K = range(2,10)
        for n_clusters in list(K):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters , affinity='euclidean' ,linkage="ward")
            preds = clusterer.fit_predict(X)
            score = silhouette_score(X, preds)
            with open("silhoute_score.txt","a+") as f:
                f.write("For n_clusters = {}, silhouette score is {}".format(n_clusters, score)+'\n')
            # print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))
        df2=pd.concat([X,y,cluster_df],axis=1)
        cluster_1=df2[df2["cluster"]==1]
        cluster_1.drop("cluster",axis=1,inplace=True)
        ##cluster 1
        cluster_1=cluster_1[cluster_1["Cover_Type"]!=3]
        cluster_1=cluster_1[cluster_1["Cover_Type"]!=4]
        rm=RandomOverSampler(random_state=30)
        X_cluster_1=cluster_1.drop("Cover_Type",axis=1)
        y_cluster_1=cluster_1["Cover_Type"]
        vf=y_cluster_1.value_counts()
        # X_cluster_1,y_cluster_1=rm.fit_resample(X_cluster_1,y_cluster_1)
        return X_cluster_1,y_cluster_1
    except Exception as e:
        raise forest_cover_exception(e,sys) from e



def model_tuning_1(X_cluster,y_cluster,base_accuracy,X_test_cluster,y_test_cluster):
    try:
        #model defined as none 
        model=None
        #defining KFOLD Cross Validation
        logging.info("KFOLD CALLED")
        kf=KFold(5)
        #applying logistics regression
        logging.info("Logistics Regression applied!!")
        log=LogisticRegression()
        #applying tuning on the parameter of Logistics Regression
        # grid_cv(log,params_g=LOGISTICS_PARAMS_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of logistics regression has been applied")
        d={'max_iter': 300,'multi_class': 'multinomial','penalty': 'l1','solver': 'saga','tol': 0.0001}
        #using logistics regression parameter
        log=LogisticRegression(penalty=d["penalty"],tol=d["tol"],random_state=40,solver=d["solver"],multi_class=d["multi_class"],max_iter=d["max_iter"])
        #applying cross val score
        scores=cross_val_score(log,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--Logistics Regression Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in logistics model
        log.fit(X_cluster,y_cluster)
        train_pred=log.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--Logistics Regression Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=log.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--Logistics Regression Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=log
                base_accuracy=model_testing_accuracy
            
        

        logging.info("Support Vector Classifier applied!!")
        svm=SVC()
        #applying tuning on the parameter of Logistics Regression
        # grid_cv(log,params_g=SVC_PARAMS_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of support vector classifier has been applied")
        d={'decision_function_shape': 'ovo','degree': 1,'gamma': 'scale','kernel': 'rbf','tol': 0.001}
        #using logistics regression parameter
        svm=SVC(kernel=d["kernel"],degree=d["degree"],gamma=d["gamma"],tol=d["tol"],decision_function_shape=d["decision_function_shape"])
        #applying cross val score
        scores=cross_val_score(svm,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--Support Vector Machine Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in logistics model
        svm.fit(X_cluster,y_cluster)
        train_pred=svm.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--Support vector classifier Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=svm.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--Support vector Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=svm
                base_accuracy=model_testing_accuracy



        

        logging.info("Decision Tree Classifier applied!!")
        dr=DecisionTreeClassifier()
        #applying tuning on the parameter of Logistics Regression
        # grid_cv(dr,params_g=DECISION_TREE_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of decision tree classifier has been applied")
        d={'criterion': 'log_loss','max_depth': 100,'max_features': 'sqrt','min_samples_split': 2,'splitter': 'best'}
        #using logistics regression parameter
        dr=DecisionTreeClassifier(criterion=d["criterion"],max_features=d["max_features"],max_depth=d["max_depth"],min_samples_split=d["min_samples_split"],splitter=d["splitter"],ccp_alpha=0.0006466247458339158,random_state=30)
        #applying cross val score
        scores=cross_val_score(dr,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--Decision Tree Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in logistics model
        dr.fit(X_cluster,y_cluster)
        train_pred=dr.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--Decision Tree Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=dr.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--Decision Tree Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=dr
                base_accuracy=model_testing_accuracy



        
        logging.info("Random Forest Classifier applied!!")
        rr=RandomForestClassifier()
        #applying tuning on the parameter of Logistics Regression
        # grid_cv(rr,params_g=RANDOM_FOREST_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of random forest classifier has been applied")
        d={'criterion': 'log_loss','max_depth': 14,'max_features': 'sqrt','min_samples_split': 2}
        #using random forest parameter
        rr=RandomForestClassifier(criterion=d["criterion"],max_depth=d['max_depth'],min_samples_split=d["min_samples_split"],max_features=d["max_features"],ccp_alpha=0.0006466247458339158,random_state=30)
        #applying cross val score
        scores=cross_val_score(rr,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--Random forest Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in logistics model
        rr.fit(X_cluster,y_cluster)
        train_pred=rr.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--Random Forest Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=rr.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--Random Forest Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=rr
                base_accuracy=model_testing_accuracy


        

        logging.info("KNeighborsClassifier applied!!")
        knn=KNeighborsClassifier()
        #applying tuning on the parameter of Logistics Regression
        # grid_cv(knn,params_g=KNN_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of KNeighborsClassifier has been applied")
        d={'algorithm': 'auto', 'leaf_size': 5, 'n_neighbors': 5, 'weights': 'distance'}
        #using random forest parameter
        knn=KNeighborsClassifier(n_neighbors=5,algorithm=d["algorithm"],leaf_size=10,weights=d["weights"])
        #applying cross val score
        scores=cross_val_score(knn,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--knn Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in logistics model
        knn.fit(X_cluster,y_cluster)
        train_pred=knn.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--KNeighborsClassifier Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=knn.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--KNeighborsClassifier Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=knn
                base_accuracy=model_testing_accuracy


        

        logging.info("AdaBoostClassifier applied!!")
        ad=AdaBoostClassifier()
        #applying tuning on the parameter of AdaBoostClassifier
        # grid_cv(ad,params_g={"base_estimator":[dr],"n_estimators":[50,100,200,300],"learning_rate":[0.001,0.0001,0.01,0.05,0.06,0.007,0.5,0.8,0.9,1.0],"algorithm":["SAMME","SAMME.R"]},X=X_cluster,y=y_cluster)
        logging.info("Tuning of AdaBoostClassifier has been applied")
        d={'algorithm': 'SAMME','base_estimator': DecisionTreeClassifier(ccp_alpha=0.0006466247458339158, criterion='log_loss',max_depth=100, max_features='sqrt', random_state=30),'learning_rate': 0.05,'n_estimators': 300}
        #using AdaBoostClassifier parameter
        ad=AdaBoostClassifier(base_estimator=d["base_estimator"],n_estimators=100,algorithm=d["algorithm"],learning_rate=0.002)
        #applying cross val score
        scores=cross_val_score(ad,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--AdaBoostClassifier Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in ad model
        ad.fit(X_cluster,y_cluster)
        train_pred=ad.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--KNeighborsClassifier Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=ad.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--KNeighborsClassifier Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=ad
                base_accuracy=model_testing_accuracy


        

        logging.info("GradientBoostingClassifier applied!!")
        gd=GradientBoostingClassifier()
        #applying tuning on the parameter of GradientBoostingClassifier
        # grid_cv(knn,params_g=GD_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of GradientBoostingClassifier has been applied")
        d={"loss":"log_loss","learning_rate":0.6,"n_estimators":100,"criterion":"friedman_mse"}
        #using GradientBoostingClassifier parameter
        gd=GradientBoostingClassifier(loss=d["loss"],learning_rate=d["learning_rate"],n_estimators=d["n_estimators"],criterion=d["criterion"])
        #applying cross val score
        scores=cross_val_score(gd,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--GradientBoostingClassifier Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in ad model
        gd.fit(X_cluster,y_cluster)
        train_pred=gd.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--GradientBoostingClassifier Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=gd.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--GradientBoostingClassifier Accuracy for testing-->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=gd
                base_accuracy=model_testing_accuracy


        return model
    except Exception as e:
        raise forest_cover_exception(e,sys) from e




    except Exception as e:
        raise forest_cover_exception(e,sys) from e


def model_tuning_2(X_cluster,y_cluster,base_accuracy,X_test_cluster,y_test_cluster):
    try:
        #model defined as none 
        model=None
        #defining KFOLD Cross Validation
        logging.info("KFOLD CALLED")
        kf=KFold(5)
        #applying logistics regression
        logging.info("Logistics Regression applied!!")
        log=LogisticRegression()
        #applying tuning on the parameter of Logistics Regression
        # grid_cv(log,params_g=LOGISTICS_PARAMS_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of logistics regression has been applied")
        d={'max_iter': 300,'multi_class': 'multinomial','penalty': 'l1','solver': 'saga','tol': 0.0001}
        #using logistics regression parameter
        log=LogisticRegression(penalty=d["penalty"],tol=d["tol"],random_state=40,solver=d["solver"],multi_class=d["multi_class"],max_iter=d["max_iter"])
        #applying cross val score
        scores=cross_val_score(log,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--Logistics Regression Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in logistics model
        log.fit(X_cluster,y_cluster)
        train_pred=log.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--Logistics Regression Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=log.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--Logistics Regression Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=log
                base_accuracy=model_testing_accuracy
            
        

        logging.info("Support Vector Classifier applied!!")
        svm=SVC()
        #applying tuning on the parameter of Logistics Regression
        # grid_cv(log,params_g=SVC_PARAMS_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of support vector classifier has been applied")
        d={'decision_function_shape': 'ovo','degree': 1,'gamma': 'scale','kernel': 'rbf','tol': 0.0001}
        #using logistics regression parameter
        svm=SVC(kernel=d["kernel"],degree=d["degree"],gamma=d["gamma"],tol=d["tol"],decision_function_shape=d["decision_function_shape"])
        #applying cross val score
        scores=cross_val_score(svm,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--Support Vector Machine Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in logistics model
        svm.fit(X_cluster,y_cluster)
        train_pred=svm.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--Support vector classifier Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=svm.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--Support vector Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=svm
                base_accuracy=model_testing_accuracy



        

        logging.info("Decision Tree Classifier applied!!")
        dr=DecisionTreeClassifier()
        #applying tuning on the parameter of Logistics Regression
        # grid_cv(dr,params_g=DECISION_TREE_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of decision tree classifier has been applied")
        d={'criterion': 'log_loss','max_depth': 100,'max_features': 'sqrt','min_samples_split': 2,'splitter': 'best'}
        #using logistics regression parameter
        dr=DecisionTreeClassifier(criterion=d["criterion"],max_features=d["max_features"],max_depth=15,min_samples_split=d["min_samples_split"],splitter=d["splitter"],ccp_alpha=0.0005614673227229467,random_state=30)
        #applying cross val score
        scores=cross_val_score(dr,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--Decision Tree Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in logistics model
        dr.fit(X_cluster,y_cluster)
        train_pred=dr.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--Decision Tree Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=dr.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--Decision Tree Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=dr
                base_accuracy=model_testing_accuracy



        
        logging.info("Random Forest Classifier applied!!")
        rr=RandomForestClassifier()
        #applying tuning on the parameter of Logistics Regression
        # grid_cv(rr,params_g=RANDOM_FOREST_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of random forest classifier has been applied")
        d={'criterion': 'log_loss','max_depth': 14,'max_features': 'sqrt','min_samples_split': 2}
        #using random forest parameter
        rr=RandomForestClassifier(criterion=d["criterion"],max_depth=9,min_samples_split=d["min_samples_split"],max_features=d["max_features"],random_state=30)
        #applying cross val score
        scores=cross_val_score(rr,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--Random forest Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in logistics model
        rr.fit(X_cluster,y_cluster)
        train_pred=rr.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--Random Forest Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=rr.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--Random Forest Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=rr
                base_accuracy=model_testing_accuracy


        

        logging.info("KNeighborsClassifier applied!!")
        knn=KNeighborsClassifier()
        #applying tuning on the parameter of Logistics Regression
        # grid_cv(knn,params_g=KNN_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of KNeighborsClassifier has been applied")
        d={'algorithm': 'auto', 'leaf_size': 5, 'n_neighbors': 1, 'weights': 'distance'}
        #using random forest parameter
        knn=KNeighborsClassifier(n_neighbors=10,algorithm="auto",leaf_size=5,weights=d["weights"])
        #applying cross val score
        scores=cross_val_score(knn,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--knn Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in logistics model
        knn.fit(X_cluster,y_cluster)
        train_pred=knn.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--KNeighborsClassifier Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=knn.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--KNeighborsClassifier Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=knn
                base_accuracy=model_testing_accuracy


        

        logging.info("AdaBoostClassifier applied!!")
        ad=AdaBoostClassifier()
        #applying tuning on the parameter of AdaBoostClassifier
        # grid_cv(ad,params_g={"base_estimator":[dr],"n_estimators":[50,100,200,300],"learning_rate":[0.001,0.0001,0.01,0.05,0.06,0.007,0.5,0.8,0.9,1.0],"algorithm":["SAMME","SAMME.R"]},X=X_cluster,y=y_cluster)
        logging.info("Tuning of AdaBoostClassifier has been applied")
        d={'algorithm': 'SAMME','base_estimator': dr,'learning_rate': 0.07,'n_estimators': 300}
        #using AdaBoostClassifier parameter
        ad=AdaBoostClassifier(base_estimator=d["base_estimator"],n_estimators=100,algorithm=d["algorithm"],learning_rate=0.002)
        #applying cross val score
        scores=cross_val_score(ad,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--AdaBoostClassifier Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in ad model
        ad.fit(X_cluster,y_cluster)
        train_pred=ad.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--KNeighborsClassifier Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=ad.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--KNeighborsClassifier Accuracy for testing--->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=ad
                base_accuracy=model_testing_accuracy


        

        logging.info("GradientBoostingClassifier applied!!")
        gd=GradientBoostingClassifier()
        #applying tuning on the parameter of GradientBoostingClassifier
        # grid_cv(knn,params_g=GD_TUNING,X=X_cluster,y=y_cluster)
        logging.info("Tuning of GradientBoostingClassifier has been applied")
        d={"loss":"log_loss","learning_rate":0.6,"n_estimators":100,"criterion":"friedman_mse"}
        #using GradientBoostingClassifier parameter
        gd=GradientBoostingClassifier(loss="log_loss",learning_rate=0.3,n_estimators=200,criterion="mse")
        #applying cross val score
        scores=cross_val_score(gd,X_cluster.values,y_cluster.values,cv=kf,scoring="f1_weighted")
        with open("cross_val_score.txt","a+") as f:
            f.write("<--GradientBoostingClassifier Cross val score--->"+"\n")
            f.write(f"scores are {scores}"+"\n")
            f.write(f"mean score is {np.mean(scores)}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        #fitting the data in ad model
        gd.fit(X_cluster,y_cluster)
        train_pred=gd.predict(X_cluster)
        model_training_accuracy=f1_score(y_cluster,train_pred,average="weighted")
        with open("Accuracy.txt","a+") as f:
            f.write("<--GradientBoostingClassifier Accuracy for training--->"+"\n")
            f.write(f"Training accuracy {model_training_accuracy}"+"\n")
            f.write(f"<--{CURRENT_TIME_STAMP}-->")
        if model_training_accuracy>=base_accuracy:
            pred=gd.predict(X_test_cluster)
            model_testing_accuracy=f1_score(y_true=y_test_cluster,y_pred=pred,average="weighted")
            with open("Accuracy.txt","a+") as f:
                f.write("<--GradientBoostingClassifier Accuracy for testing-->"+"\n")
                f.write(f"Testing accuracy {model_testing_accuracy}"+"\n")
                f.write(f"<--{CURRENT_TIME_STAMP}-->")
            if model_testing_accuracy>=base_accuracy:
                model=gd
                base_accuracy=model_testing_accuracy


        return model
    except Exception as e:
        raise forest_cover_exception(e,sys) from e





        
        




        






        




        






        


    except Exception as e:
        raise forest_cover_exception(e,sys) from e


        


        











            


    


    


    






