import time; start_time = time.time()
import numpy as np
import pandas as pd
from sklearn import ensemble
import xgboost as xgb
from sklearn.metrics import log_loss, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GMM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, Ridge
#from enn import ENN
from scipy.spatial.distance import euclidean, mahalanobis
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from lightning.classification import CDClassifier, FistaClassifier, SAGAClassifier, LinearSVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
#from pyearth import Earth
from sklearn.cluster import k_means
from elm import ELMClassifier, ELMRegressor, GenELMClassifier, GenELMRegressor
from random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer
import random; random.seed(2016)
import sys

d=4
#d=int(sys.argv[2]) #d external parameter

print("file: "+str(d)+"\n")

train = pd.read_csv('input/numerai_training_data_'+str(d)+'.csv', encoding="utf-8-sig") 
test = pd.read_csv('input/numerai_tournament_data_'+str(d)+'.csv', encoding="utf-8-sig")

num_train = train.shape[0]

y_train = train['target']
train = train.drop(['target'],axis=1)
id_test = test['t_id']

def fill_nan_null(val):
    ret_fill_nan_null = 0.0
    if val == True:
        ret_fill_nan_null = 1.0
    return ret_fill_nan_null

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['null_count'] = df_all.isnull().sum(axis=1).tolist()
df_all_temp = df_all['t_id']
df_all = df_all.drop(['t_id'],axis=1)
df_data_types = df_all.dtypes[:] #{'object':0,'int64':0,'float64':0,'datetime64':0}
d_col_drops = []

df_all = pd.concat([df_all, df_all_temp], axis=1)
print(len(df_all), len(df_all.columns))
train = train.drop(d_col_drops,axis=1)
test = test.drop(d_col_drops,axis=1)

def flog_loss(ground_truth, predictions):
    flog_loss_ = log_loss(ground_truth, predictions) #, eps=1e-15, normalize=True, sample_weight=None)
    return flog_loss_
LL  = make_scorer(flog_loss, greater_is_better=False)

#m=int(sys.argv[1]) #m external parameter
m=1
print("m: "+str(m)+"\n")

#mf - max features

if m==1:
    g={'ne':150,'md':6,'mf':21+d,'rs':2016}  #change to gne':500,'md':40,'mf':60,'rs':2016}
elif m==2:
    g={'ne':500,'md':40,'mf':21+d,'rs':2016}
elif m==3:
    g={'ne':300,'md':6,'mf':21+d,'rs':2016}
elif m==4:
    g={'ne':150,'md':5,'mf':21+d,'rs':2016}
elif m==5:
    g={'ne':300,'md':4,'mf':21+d,'rs':2016}
elif m==6:
    g={'ne':300,'md':8,'mf':21+d,'rs':2016}
elif m==7:
    g={'ne':150,'md':10,'mf':21+d,'rs':2016}
elif m==8:
    g={'ne':400,'md':8,'mf':15+d,'rs':2016}
elif m==9:
    g={'ne':500,'md':9,'mf':15+d,'rs':2016}
elif m==10:
    g={'ne':500,'md':10,'mf':20+d,'rs':2016}
elif m==11:
    g={'ne':400,'md':7,'mf':21+d,'rs':2016}


#logistic regression 
lg1c = LogisticRegression(solver = 'liblinear', penalty = 'l1', C = 1.0, n_jobs = -1)
lg2c = LogisticRegression(solver = 'liblinear', penalty = 'l1', C = .01, n_jobs = -1)    
lg3c = LogisticRegression(solver = 'liblinear', penalty = 'l1', C = 100, n_jobs = -1)    
lg4c = LogisticRegression(solver = 'liblinear', penalty = 'l2', C = 1.0, n_jobs = -1)   #C=1.0 default
lg5c = LogisticRegression(solver = 'liblinear', penalty = 'l2', C = .01, n_jobs = -1)    
lg6c = LogisticRegression(solver = 'liblinear', penalty = 'l2', C = 100, n_jobs = -1)     
#
#lg7c = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, solver='sag', max_iter=10000, n_jobs=-1)
lg7c = LogisticRegression(solver = 'liblinear', penalty = 'l1', C = 10, n_jobs = -1)
lg8c = LogisticRegression(solver = 'liblinear', penalty = 'l2', C = 10, n_jobs = -1)

#ridge
ridge1 = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001)
ridge2 = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='svd', tol=0.001)
ridge3 = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='cholesky', tol=0.001)
ridge4 = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='lsqr', tol=0.001)
ridge5 = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='sparse_cg', tol=0.001)
ridge6 = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='sag', tol=0.001)
#ridge7 = Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=10000,normalize=False, random_state=None, solver='auto', tol=0.001)

#lda,qda
ldac = LinearDiscriminantAnalysis()
qdac = QuadraticDiscriminantAnalysis()

#knn
knn1c = KNeighborsClassifier(n_neighbors=25, p = 1, weights = 'distance',n_jobs=-1)
knn2c = KNeighborsClassifier(n_neighbors=25, p = 2, weights = 'distance',n_jobs=-1 )
knn3c = KNeighborsClassifier(n_neighbors=25, metric = 'chebyshev', weights = 'distance',n_jobs=-1)

nc1 = NearestCentroid()

#semi-supervised
#LabelPropagation
#LabelSpreading

#perceptron
per1 = Perceptron(n_jobs=-1, n_iter=10)
per2 = Perceptron(penalty='l2', n_jobs=-1, n_iter=10)
per3 = Perceptron(penalty='l1', n_jobs=-1, n_iter=10)
per4 = Perceptron(penalty='elasticnet', n_jobs=-1, n_iter=10)

#passive aggresive
passaggr = PassiveAggressiveClassifier(n_jobs=-1, n_iter=10)

#lightning
CD1_c = CDClassifier(loss='log', penalty='l2', multiclass=False, C=1.0, alpha=1.0, max_iter=50, tol=0.001, termination='violation_sum', shrinking=True, max_steps='auto', sigma=0.01, beta=0.5, warm_start=False, debiasing=False, Cd=1.0, warm_debiasing=False, selection='cyclic', permute=True, callback=None, n_calls=100, random_state=None, verbose=1)
CD2_c = CDClassifier(loss='modified_huber', penalty='l2', multiclass=False, C=1.0, alpha=1.0, max_iter=50, tol=0.001, termination='violation_sum', shrinking=True, max_steps='auto', sigma=0.01, beta=0.5, warm_start=False, debiasing=False, Cd=1.0, warm_debiasing=False, selection='cyclic', permute=True, callback=None, n_calls=100, random_state=None, verbose=1)
SAGA_c = SAGAClassifier(eta='auto', alpha=1.0, beta=0.0, loss='modified_huber', penalty=None, gamma=1.0, max_iter=100, n_inner=1.0, tol=0.001, verbose=0, callback=None, random_state=None)

#Fista1 = FistaClassifier(C=1.0, alpha=1.0, loss='log', penalty='l2', multiclass=False, max_iter=100, max_steps=30, eta=2.0, sigma=1e-05, callback=None, verbose=1)
#Fista2_c = FistaClassifier(C=1.0, alpha=1.0, loss='modified_huber', penalty='l1', multiclass=False, max_iter=100, max_steps=30, eta=2.0, sigma=1e-05, callback=None, verbose=0)
#SAGA_c = SAGAClassifier(eta='auto', alpha=1.0, beta=0.0, loss='modified_huber', penalty=None, gamma=1.0, max_iter=100, n_inner=1.0, tol=0.001, verbose=0, callback=None, random_state=None)
#LinSVC_c = LinearSVC(C=1.0, loss='hinge', criterion='accuracy', max_iter=1000, tol=0.001, permute=True, shrinking=True, warm_start=False, random_state=None, callback=None, n_calls=100, verbose=0)

#earth
#earth_c = Earth()

elm1 = ELMClassifier(n_hidden=500, activation_func='multiquadric')
elm2 = ELMRegressor(random_state=0, activation_func='gaussian', alpha=0.1)
elm3 = ELMClassifier(n_hidden=1000, activation_func='gaussian', alpha=0.1, random_state=0)
elm4 = ELMClassifier(n_hidden=500, activation_func='hardlim', alpha=0.1, random_state=0)
elm5 = ELMRegressor(random_state=0, activation_func='tanh', alpha=0.1)
elm6 = ELMRegressor(random_state=0, activation_func='sine', alpha=0.1)
elm7 = ELMRegressor(random_state=0, activation_func='tribas', alpha=0.1)
elm8 = ELMRegressor(random_state=0, activation_func='inv_tribas', alpha=0.1)
elm9 = ELMRegressor(random_state=0, activation_func='sigmoid', alpha=0.1)
elm10 = ELMRegressor(random_state=0, activation_func='hardlim', alpha=0.1)
elm11 = ELMRegressor(random_state=0, activation_func='softlim', alpha=0.1)
elm12 = ELMRegressor(random_state=0, activation_func='multiquadric', alpha=0.1)
elm13 = ELMRegressor(random_state=0, activation_func='inv_multiquadric', alpha=0.1)
#'tanh', 'sine', 'tribas', 'inv_tribase', 'sigmoid','hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric'

#orig        
etc = ensemble.ExtraTreesClassifier(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], criterion='entropy', min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)      
etr = ensemble.ExtraTreesRegressor(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)      
rfc = ensemble.RandomForestClassifier(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], criterion='entropy', min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)
rfr = ensemble.RandomForestRegressor(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)
xgr = xgb.XGBRegressor(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=0.02, subsample=0.9, colsample_bytree=0.85, objective='reg:linear')
xgc = xgb.XGBClassifier(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=0.02, subsample=0.9, colsample_bytree=0.85, objective='binary:logistic') #try 'binary:logitraw'


#clf = {'elm1':elm1,'elm2':elm2,'elm3':elm3,'elm4':elm4,'elm5':elm5,'elm6':elm6,
#       'elm7':elm7,'elm8':elm8,'elm9':elm9,'elm10':elm10,'elm11':elm11,'elm12':elm12,'elm13':elm13}

#clf = {'nc1':nc1}
#clf = {'SAGA_c':SAGA_c}#,'Fista2_c':Fista2_c}
#clf = {'earth_c':earth_c}

#clf = {'CD1_c':CD1_c,'CD2_c':CD2_c}
#clf = {'ridge1':ridge1,'ridge2':ridge2,'ridge3':ridge3,'ridge4':ridge4,'ridge5':ridge5,'ridge6':ridge6}

#clf = {'per1':per1, 'per2':per2,'per3':per3,'per4':per4,'passagr':passaggr}
#clf = {'knn1c':knn1c,'knn2c':knn2c,'knn3c':knn3c}
#clf = {'ldac':ldac,'qdac':qdac}

#final set of classifiers:
clf = {'lg1c':lg1c,'lg2c':lg2c,'lg3c':lg3c,'lg4c':lg4c,'lg5c':lg5c,'lg6c':lg6c,'lg7c':lg7c,'lg8c':lg8c} 
#       'etc':etc, 'etr':etr, 'rfc':rfc, 'rfr':rfr, 'xgr':xgr, 'xgc':xgc} 

y_pred=[]
best_score = 0.0
id_results = id_test[:]
for c in clf:
    if c[:1] != "x": #not xgb
        model = GridSearchCV(estimator=clf[c], param_grid={}, n_jobs =-1, cv=2, verbose=0, scoring=LL)    
        model.fit(train, y_train.values)
        if c[-1:] != "c": #not classifier
            y_pred = model.predict(test)
            if c[:3] == "elm":
               print("Ensemble Model: ", c, " Best CV score: ", model.best_score_)#, " Time: ", round(((time.time() - start_time)/60),2))
            else:
               print("Ensemble Model: ", c, " Best CV score: ", model.best_score_, " Time: ", round(((time.time() - start_time)/60),2)) 
        else: #classifier
            best_score = (log_loss(y_train.values, model.predict_proba(train)))*-1
            y_pred = model.predict_proba(test)[:,1]
            print("Ensemble Model: ", c, " Best CV score: ", best_score, " Time: ", round(((time.time() - start_time)/60),2))
    else: #xgb
        X_fit, X_eval, y_fit, y_eval= train_test_split(train, y_train, test_size=0.35, train_size=0.65, random_state=g['rs'])
        model = clf[c]
        model.fit(X_fit, y_fit.values, early_stopping_rounds=20, eval_metric="logloss", eval_set=[(X_eval, y_eval)], verbose=0)
        if c == "xgr": #xgb regressor
            best_score = (log_loss(y_train.values, model.predict(train)))*-1
            y_pred = model.predict(test)
        else: #xgb classifier
            best_score = (log_loss(y_train.values, model.predict_proba(train)))*-1
            y_pred = model.predict_proba(test)[:,1]
        print("Ensemble Model: ", c, " Best CV score: ", best_score, " Time: ", round(((time.time() - start_time)/60),2))
    for i in range(len(y_pred)):
        if y_pred[i]<0.0:
            y_pred[i] = 0.0
        if y_pred[i]>1.0:
            y_pred[i] = 1.0
    df_in = pd.DataFrame({"t_id": id_test, c: y_pred})
    id_results = pd.concat([id_results, df_in[c]], axis=1)

id_results['avg'] = id_results.drop('t_id', axis=1).apply(np.average, axis=1)
id_results['min'] = id_results.drop('t_id', axis=1).apply(min, axis=1)
id_results['max'] = id_results.drop('t_id', axis=1).apply(max, axis=1)
id_results['diff'] = id_results['max'] - id_results['min']
#for i in range(10):
#    print(i, len(id_results[id_results['diff']>(i/10)]))
id_results.to_csv("output/results_analysis_"+str(d)+"_"+str(m)+".csv", index=False)
ds = id_results[['t_id','avg']]
ds.columns = ["t_id","probability"]
ds.to_csv("output/submission_"+str(d)+"_"+str(m)+".csv",index=False)

#import xgbfir
#xgbfir.saveXgbFI(xgc, OutputXlsxFile = "ouput/xgbfir_"+str(d)+"_"+str(m)+".xlsx")
