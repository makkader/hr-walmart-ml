#Author: Md Abdul Kader
#email: abdul.kader880@gmail.com
#University of Texas at El Paso
#https://www.linkedin.com/in/makkader

import numpy as np;
import csv;
from collections import defaultdict;
from sklearn import tree;
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier;
from sklearn.naive_bayes import *;
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer;
from bs4 import BeautifulSoup;
import re;
from sklearn.ensemble import RandomForestClassifier;
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn import svm;
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.decomposition import PCA,RandomizedPCA
from sklearn.linear_model import *;
from sklearn.svm import *;
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import *;
from sklearn import preprocessing;
import xgboost as xgb;
from xgboost.sklearn import XGBClassifier;
from sklearn.decomposition import *;


def getListOfShelfID(s):
    return [int(a) for a in s.strip()[1:-1].split(',')];


def remove_urls (text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text, flags=re.MULTILINE)
    return text;
def getClearString(text):
    if text==".":
        return "";
    text=remove_urls(text);
    text=re.sub(r'[^\x00-\x7F]+',' ', text);
    text=BeautifulSoup(text,'html.parser').get_text(" ");
    text=text.replace("/"," ");
    text=text.replace("\\"," ");
    text=text.replace("."," ");

    return text.encode("utf8");
def readCSV(path):


    ll=[];
    y=[];

    csvreader=csv.reader(open(path),delimiter=',');

    c=0;
    for line in csvreader:
        c+=1;
        if (c==1):
            #print "header skipping";
            continue;

        for i in xrange(len(line)):
            line[i]=getClearString(line[i]).strip().lower();


        ll.append(line);
        #print line;
        if QUICK==True:
            if c>100:
                break;


    return np.array(ll);

def splitIntoXandY(dataTrain):
    X=dataTrain[:,:-1];
    y=dataTrain[:,-1];
    Y=[]
    for shelflist in y:
        Y.append(getListOfShelfID(shelflist));
    return X,np.array(Y);

def getMaps(Y):
    shelfID2classID={};
    classID2shelfID={};

    for shelflist in Y:
        for shelfid in shelflist:
            if shelfid not in shelfID2classID:
                cid=len(shelfID2classID);

                shelfID2classID[shelfid]=cid;
                classID2shelfID[cid]=shelfid;

    return shelfID2classID,classID2shelfID;

def getYMat(Y,shelfID2classID,classID2shelfID):
    n=len(Y);
    m=len(shelfID2classID);
    YMat=np.zeros( (n,m),dtype=int );
    i=0;
    for shelflist in Y:

        for shelfid in shelflist:
            cid=shelfID2classID[shelfid];
            YMat[i][cid]=1;
        i+=1;
    return YMat;

def getProbabilityFeatureForSINGLEColumn(XTraini,YMat,XTesti):

    clf = Pipeline([('vect', CountVectorizer(stop_words="english")),('tfidf', TfidfTransformer(use_idf=True)),('clf', OneVsRestClassifier(LogisticRegression(class_weight='balanced')) ),]);
    clf.fit(XTraini,YMat);

    proTrain=clf.predict_proba(XTraini);
    proTest=clf.predict_proba(XTesti);

    return proTrain,proTest;

def getColumnwiseIndivitualProbabilityFeature(listOfColumns,XTrain,YMat,XTest):
    proAllTrain=None;
    proAllTest=None;

    for col in listOfColumns:
        proTrain,proTest=getProbabilityFeatureForSINGLEColumn(XTrain[:,col],YMat,XTest[:,col]);
        if proAllTrain==None:
            proAllTrain=proTrain;
            proAllTest=proTest;
        else:
            proAllTrain=np.concatenate((proAllTrain,proTrain),axis=1);
            proAllTest=np.concatenate((proAllTest,proTest),axis=1);

    return proAllTrain,proAllTest;


def revisedProbabilityFeature(pro,Xi):
    for i in xrange(Xi.shape[0]):
        if not Xi[i]:
            pro[i].fill(0);
    return pro;

def getCombinedX(colsForTextClassification,X):
    combinedTextCol=[];
    for i in xrange(X.shape[0]):
        txt=""
        for col in colsForTextClassification:
            txt=txt+" "+X[i,col];


        combinedTextCol.append(txt);

    return combinedTextCol;

def getTFIDFFeatureForListOfColumns(listOfColumns,XTrain,XTest):

    cumulativeTextTrain=getCombinedX(listOfColumns,XTrain);
    cumulativeTextTest=getCombinedX(listOfColumns,XTest);
    #print np.array(tx).shape;
    vectorizer = Pipeline([('vect', CountVectorizer(stop_words="english",max_features=5000)),('tfidf', TfidfTransformer(use_idf=True)),]);
    vectorizer.fit(cumulativeTextTrain);

    TFIDFTrain=vectorizer.transform(cumulativeTextTrain);
    TFIDFTest=vectorizer.transform(cumulativeTextTest);

    #dimension reduction
    svd = TruncatedSVD(n_components=200, n_iter=12, random_state=42);
    svd.fit(TFIDFTrain);
    #plt.plot(svd.explained_variance_ratio_);
    #plt.show();

    TFIDFTrain=svd.transform(TFIDFTrain);
    TFIDFTest=svd.transform(TFIDFTest);
    #normalize
    norma=preprocessing.Normalizer().fit(TFIDFTrain);
    TFIDFTrain=norma.transform(TFIDFTrain)
    TFIDFTest=norma.transform(TFIDFTest);


    return TFIDFTrain,TFIDFTest;


def onHotEncoderValue2Idx(X_tr_i,X_tst_i):
    value2Idx={};
    for i in xrange(X_tr_i.shape[0]):
        term=X_tr_i[i]
        if term not in value2Idx:
            value2Idx[term]=len(value2Idx);

    for i in xrange(X_tst_i.shape[0]):
        term=X_tst_i[i]
        if term not in value2Idx:
            value2Idx[term]=len(value2Idx);

    return value2Idx;

def onHotEncoderCol2Value2Idx(X_tr,X_tst,colList):
    col2Value2Idx={};
    for col in colList:
        col2Value2Idx[col]=onHotEncoderValue2Idx(X_tr[:,col],X_tst[:,col]);
    return col2Value2Idx;

def getOnHotEncoding(Xi,value2Idx):
    m=len(value2Idx);
    X=np.zeros((Xi.shape[0],m));

    for i in xrange(Xi.shape[0]):
        col=value2Idx[Xi[i]];
        X[i,col]=1;
    return X;

def getCategoricalFeatures(listOfCols,col2Value2Idx,XTrain,XTest):
    featureTrain=None;
    featureTest=None;
    for col in listOfCols:
        tr=getOnHotEncoding(XTrain[:,col],col2Value2Idx[col]);
        tst=getOnHotEncoding(XTest[:,col],col2Value2Idx[col]);

        if featureTrain==None:
            featureTrain=tr;
            featureTest=tst;
        else:

            featureTrain=np.concatenate((featureTrain,tr),axis=1);
            featureTest=np.concatenate((featureTest,tst),axis=1);



    #reduce dimension of categorical feature;
    pca=PCA(n_components=20);
    pca.fit(featureTrain);
    #plt.plot(pca.explained_variance_ratio_);
    #plt.show();
    #sys.exit();

    featureTrain=pca.transform(featureTrain);
    featureTest=pca.transform(featureTest);


    #normalize
    norma=preprocessing.Normalizer();
    featureTrain=norma.transform(featureTrain)
    featureTest=norma.transform(featureTest);

    return featureTrain,featureTest;

def getZeroOneFeature(Xi):
    X=np.zeros((Xi.shape[0],1));

    for i in xrange(Xi.shape[0]):
        if Xi[i]:
            X[i,0]=1;
    return X;
def getZeroOneFeatureMatrix(colZeroOneFeature,data):
    X=np.zeros((data.shape[0],0));

    for col in colZeroOneFeature:
        t=getZeroOneFeature(data[:,col]);
        X=np.concatenate((X,t),axis=1);
    return X;

def getCombinedProbFeature(colList,X_tr,YMat,X_tst):

    combinedText_train=getCombinedX(colList,X_tr);
    combinedText_test=getCombinedX(colList,X_tst);

    clf = Pipeline([('vect', CountVectorizer(stop_words="english")),('tfidf', TfidfTransformer(use_idf=True)),('clf', OneVsRestClassifier(LogisticRegression(class_weight='balanced')) ),]);
    clf.fit(combinedText_train,YMat);

    pro_tr=clf.predict_proba(combinedText_train);
    pro_tst=clf.predict_proba(combinedText_test);


    clf2 = Pipeline([('vect', CountVectorizer(stop_words="english")),('tfidf', TfidfTransformer(use_idf=False)),('clf', OneVsRestClassifier(MultinomialNB(alpha=1.0)) ),]);
    clf2.fit(combinedText_train,YMat);
    pro_tr2=clf2.predict_proba(combinedText_train);
    pro_tst2=clf2.predict_proba(combinedText_test);

    pro_tr=np.concatenate((pro_tr,pro_tr2),axis=1);
    pro_tst=np.concatenate((pro_tst,pro_tst2),axis=1);

    return pro_tr,pro_tst;



def myValidation(NumericX,YMat):
    print "in My Validation:"

    #NumericX=NumericX[0:2000,:];
    #YMat=YMat[0:2000,:];

    clf=OneVsRestClassifier(LinearSVC(class_weight='balanced',loss='squared_hinge'));
    scores = cross_val_score(clf, NumericX, YMat, cv=3,scoring="f1_micro");
    print np.mean(scores);


def getStackingFeatures(NumericTrain,YMat,NumericTest):
    norma=preprocessing.Normalizer().fit(NumericTrain);
    NumericTrain=norma.transform(NumericTrain);
    NumericTest=norma.transform(NumericTest);

    defaultCLF = OneVsRestClassifier(LogisticRegression(class_weight='balanced'));
    defaultCLF.fit(NumericTrain,YMat);
    proTrain=defaultCLF.predict_proba(NumericTrain);
    proTest=defaultCLF.predict_proba(NumericTest);

    return proTrain,proTest;

def getItemIDClassProbFeature(NumericTrain,itemClassID9,NumericTest):
    norma=preprocessing.Normalizer().fit(NumericTrain);
    NumericTrain=norma.transform(NumericTrain);
    NumericTest=norma.transform(NumericTest);

    defaultCLF = LogisticRegression(class_weight='balanced');
    defaultCLF.fit(NumericTrain,itemClassID9);
    proTrain=defaultCLF.predict_proba(NumericTrain);
    proTest=defaultCLF.predict_proba(NumericTest);

    #it works
    defaultCLF2 = RandomForestClassifier(n_estimators=60,class_weight='balanced');
    defaultCLF2.fit(NumericTrain,itemClassID9);
    proTrain2=defaultCLF2.predict_proba(NumericTrain);
    proTest2=defaultCLF2.predict_proba(NumericTest);


    proTrain=np.concatenate((proTrain,proTrain2),axis=1);
    proTest=np.concatenate((proTest,proTest2),axis=1);

    return proTrain,proTest;

def getListOfPredictors():
    s=42;
    defaultCLF0 = OneVsRestClassifier(XGBClassifier()); #slow but best;
    defaultCLF1 = OneVsRestClassifier(SGDClassifier(alpha=.0001, n_iter=50,penalty="l2"))#2nd best

    rf = RandomForestClassifier(n_estimators=60,class_weight='balanced',random_state=s);
    lr = LogisticRegression(class_weight='balanced');
    etc= ExtraTreesClassifier(n_estimators=70);

    #soft voting;
    eclf=OneVsRestClassifier(VotingClassifier(estimators=[('clf1',etc ),('clf2',lr ),('clf3',rf )], voting='soft'));

    return [eclf,defaultCLF1,defaultCLF0];


def fitPredictors(listOfPredictors,X,YMat):
    print "Fitting predictor models";
    for clf in listOfPredictors:
        clf.fit(X,YMat);
        print "clf#";

def predictTest(listOfPredictors,testX):
    listOfYhat=[]
    for clf in listOfPredictors:
        yhat=clf.predict(testX);
        listOfYhat.append(yhat);

    sp=listOfYhat[0].shape;
    yhat=np.zeros(sp,dtype=int);
    for i in xrange(sp[0]):
        for j in xrange(sp[1]):
            vote=0;
            for ahat in listOfYhat:
                vote+=ahat[i,j];
            if vote>= len(listOfYhat)-vote:
                yhat[i,j]=1;
    return yhat;

def fun(X,Y,dataTest):

    shelfID2classID,classID2shelfID=getMaps(Y);
    YMat=getYMat(Y,shelfID2classID,classID2shelfID);


    #feature set 1
    colZeroOneFeature=[2,4,5,6,7,8,10]
    #feature set 2
    colsForCategoricalFeature=[5,6,7,9,10,11,16,17,18,21];
    #feature set 3
    colsForCom=range(1,22)#[2,12,13,14,19,20]

    #feature set 4
    colsForIndividualProb=[13,19,9,14]


    #feature set 5
    colSubsetCombined4Prob=[3,12,13,14,19,9]#,15,16,17,18,20];


    #initial empty numeric training and testing mat
    NumericTrain=np.zeros((X.shape[0],0));
    NumericTest=np.zeros((dataTest.shape[0],0));


    #Probability feature for combined columns
    pro_tr,pro_tst=getCombinedProbFeature(colSubsetCombined4Prob,X,YMat,dataTest);
    NumericTrain=np.concatenate((NumericTrain, pro_tr), axis=1);
    NumericTest=np.concatenate((NumericTest, pro_tst), axis=1);

    #zero one feature
    NumericTrain=np.concatenate((NumericTrain, getZeroOneFeatureMatrix(colZeroOneFeature,X)), axis=1);
    NumericTest=np.concatenate((NumericTest, getZeroOneFeatureMatrix(colZeroOneFeature,dataTest)), axis=1);


    #TFIDF feature for all columns
    TFIDFTrain,TFIDFTest=getTFIDFFeatureForListOfColumns(colsForCom,X,dataTest);
    NumericTrain=np.concatenate((NumericTrain, TFIDFTrain), axis=1);
    NumericTest=np.concatenate((NumericTest, TFIDFTest), axis=1);

    #columnwise probability features
    proTrain,proTest=getColumnwiseIndivitualProbabilityFeature(colsForIndividualProb,X,YMat,dataTest);
    NumericTrain=np.concatenate((NumericTrain, proTrain), axis=1);
    NumericTest=np.concatenate((NumericTest, proTest), axis=1);


    #categorical
    col2Value2Idx=onHotEncoderCol2Value2Idx(X,dataTest,colsForCategoricalFeature);
    featureTrain,featureTest=getCategoricalFeatures(colsForCategoricalFeature,col2Value2Idx,X,dataTest);
    NumericTrain=np.concatenate((NumericTrain, featureTrain), axis=1);
    NumericTest=np.concatenate((NumericTest, featureTest), axis=1);


    #probability for itemclass id, col9
    proTrain,proTest=getItemIDClassProbFeature(NumericTrain,X[:,9],NumericTest)
    NumericTrain=np.concatenate((NumericTrain, proTrain), axis=1);
    NumericTest=np.concatenate((NumericTest, proTest), axis=1);

    #stacking
    proTrain,proTest=getStackingFeatures(NumericTrain,YMat,NumericTest);
    NumericTrain=np.concatenate((NumericTrain, proTrain), axis=1);
    NumericTest=np.concatenate((NumericTest, proTest), axis=1);

    #row wise normalization
    norma=preprocessing.Normalizer().fit(NumericTrain);
    NumericTrain=norma.transform(NumericTrain);
    NumericTest=norma.transform(NumericTest);

    print "Training shape: ",NumericTrain.shape;
    print "Test shape: ",NumericTest.shape;


    #############***********my validation*********#####
    if VALIDATION==True:
        myValidation(NumericTrain,YMat);
        return;

    #for prediction on testset
    goForPrediction(NumericTrain,YMat,NumericTest,dataTest[:,0],classID2shelfID);


def goForPrediction(NumericTrain,YMat,NumericTest,itemIDList,classID2shelfID):
    listOfPredictors=getListOfPredictors();
    fitPredictors(listOfPredictors,NumericTrain,YMat);
    yhat=predictTest(listOfPredictors,NumericTest);

    yhat=testsetWithZeroPrediction(yhat,NumericTrain,YMat,NumericTest,listOfPredictors[0]);


    print "yhat shape: ",yhat.shape;
    generateOutputFile(itemIDList,yhat,classID2shelfID);
    print "Output file (tags.tsv) is generated successfully."


def testsetWithZeroPrediction(yhat,NumericTrain,YMat,NumericTest,clf):
    emptyPredictionRowList=[];

    for i in xrange(yhat.shape[0]):
        noPred=np.sum(yhat[i,:]);
        if noPred==0:
            emptyPredictionRowList.append(i);

    NumericTestForEmpty=NumericTest[emptyPredictionRowList,:];


    yhatForEmpty=clf.predict_proba(NumericTestForEmpty);
    maxProbClassIDList=np.argmax(yhatForEmpty, axis=1)


    #print yhatForEmpty.shape;

    for i in xrange(len(emptyPredictionRowList)):
        rowid=emptyPredictionRowList[i];
        yhat[rowid,maxProbClassIDList[i]]=1;
    return yhat;


def getReducedMatrix(NumericTrain,NumericTest,ncom):
    #return NumericTrain,NumericTest;

    pca=PCA(n_components=ncom);
    pca.fit(NumericTrain);
    #plt.plot(pca.explained_variance_ratio_[50:]);
    #plt.show();
    #sys.exit();

    NumericTrain=pca.transform(NumericTrain);
    NumericTest=pca.transform(NumericTest);


    #normalize
    norma=preprocessing.Normalizer();
    NumericTrain=norma.transform(NumericTrain)
    NumericTest=norma.transform(NumericTest);

    return NumericTrain,NumericTest;


def generateOutputFile(itemIDList,yhat,classID2shelfID):

    itemid2shelfIdList=defaultdict(list);
    for i in xrange(yhat.shape[0]):
        itemid=int(itemIDList[i]);
        predictedShelfList=[]
        for j in xrange(yhat.shape[1]):
            if yhat[i,j]==1:
                predictedShelfList.append(str(classID2shelfID[j]));
        itemid2shelfIdList[itemid]=predictedShelfList;


    fout=open("tags.tsv","w");

    fout.write("item_id\ttag\n");
    for itemid in sorted(itemid2shelfIdList.keys()):
        predictedShelfList=itemid2shelfIdList[itemid];
        fout.write(str(itemid)+"\t["+','.join(predictedShelfList)+"]\n");

    fout.close();



#main method

#data file paths converted to csv from tsv using MS excel
trainPath="train.csv";
testPath="test.csv";


#parameters for quick validation
QUICK=False;
VALIDATION=False;



dataTrain=readCSV(trainPath);
print "Trainingset reading done!"
dataTest=readCSV(testPath);
print "Testset reading done!"

XTrain,YTrain=splitIntoXandY(dataTrain);
fun(XTrain,YTrain,dataTest);



