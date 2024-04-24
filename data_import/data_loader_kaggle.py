from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import manifold, decomposition
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import offsetbox
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
import pandas as pd
import random
import sqlite3
import datetime
import matplotlib.cm as cmx
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.decomposition import KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import svm
import seaborn as sns
from sklearn import metrics


import os.path
'''
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "database.sqlite")

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect(db_path)

year_greater = 2009 #we have no team attribute data before 2010

df_TeamAttributes = pd.read_sql_query("SELECT * from Team_Attributes", con)
df_Match = pd.read_sql_query("SELECT * from Match", con)
#df = df.loc[:, (df.columns != 'id' or df.columns != 'team_fifa_api_id')]
#df = df.drop(['id','team_fifa_api_id','team_api_id'], axis=1)
df_TeamAttributes['date'] = pd.to_datetime(df_TeamAttributes['date'])
df_Match['date'] = pd.to_datetime(df_TeamAttributes['date'])
df_Match = df_Match[df_Match.date.dt.year > year_greater] #drop all data we have no team attributes for
#df_Data = pd.DataFrame()
tmp_attr = []
tmp_match = []
tmp_attr_away = []
tmp_match_away = []

for i in range(0,df_Match.shape[0]):
    yearfind = pd.to_datetime(df_Match.iloc[i].date).year
    team_id = df_Match.iloc[i].home_team_api_id
    away_team_id = df_Match.iloc[i].away_team_api_id
    result = df_TeamAttributes[(df_TeamAttributes['team_api_id']==team_id) & (df_TeamAttributes['date'].dt.year==yearfind)]
    result_away = df_TeamAttributes[(df_TeamAttributes['team_api_id']==away_team_id) & (df_TeamAttributes['date'].dt.year==yearfind)]
    if not any((result.empty,result_away.empty)):
        tmp_attr.append(result)
        tmp_match.append(df_Match.iloc[[i]])
        tmp_attr_away.append(result_away)




#########HOME##########
tmp_attr = pd.concat(tmp_attr)
tmp_attr = tmp_attr.reset_index(drop=True)
tmp_match = pd.concat(tmp_match)
tmp_match = tmp_match.reset_index(drop=True)
tmp_match = tmp_match[['home_team_goal', 'away_team_goal']]
df_Data = pd.concat([tmp_attr,tmp_match],axis=1)
drop_list = ['id', 'team_fifa_api_id', 'team_api_id', 'date', 'buildUpPlayDribbling'] #too many NA in dribbling
df_Data = df_Data.drop(drop_list, axis=1)
#df_Data = df_Data.dropna()


    
#encode strings
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = df_Data.select_dtypes(include=['object']).columns.tolist()
int_columns = df_Data.select_dtypes(include=['int']).columns.tolist()
df_encoded = df_Data[categorical_columns].apply(preprocessing.LabelEncoder().fit_transform)
data_encoded = pd.concat([df_Data[int_columns], df_encoded], axis=1)
df_Data = data_encoded
df_Data_home = df_Data
##########################

########AWAY##############
tmp_attr_away = pd.concat(tmp_attr_away)
tmp_attr_away = tmp_attr_away.reset_index(drop=True)
df_Data = pd.concat([tmp_attr_away,tmp_match],axis=1)
drop_list = ['id', 'team_fifa_api_id', 'team_api_id', 'date', 'buildUpPlayDribbling'] #too many NA in dribbling
df_Data = df_Data.drop(drop_list, axis=1)
#df_Data = df_Data.dropna()
    
#encode strings
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = df_Data.select_dtypes(include=['object']).columns.tolist()
int_columns = df_Data.select_dtypes(include=['int']).columns.tolist()
df_encoded = df_Data[categorical_columns].apply(preprocessing.LabelEncoder().fit_transform)
data_encoded = pd.concat([df_Data[int_columns], df_encoded], axis=1)
df_Data = data_encoded
df_Data_away = df_Data

df_Data = df_Data_home - df_Data_away
###########################

df_Data = pd.concat([df_Data,tmp_match],axis=1)
df_Data = df_Data.dropna()

#print(df_Data.head())

con.close()

y = []
for i in range(0,df_Data.shape[0]):
    home_goals = pd.to_numeric(df_Data.iloc[i].home_team_goal)
    away_goals = pd.to_numeric(df_Data.iloc[i].away_team_goal)
    win = home_goals-away_goals
    if win > 0:
        win = 1
    elif win < 0:
        win = -1
    else:
        win = -1
    y.append(win)


X = df_Data.drop(['home_team_goal','away_team_goal'],axis=1)
y = pd.DataFrame(y,columns=['Win'])
y = pd.to_numeric(y['Win'])


#df_Data.to_excel("encoded.xlsx")
#df_Data.to_csv('encoded.csv')
'''

df_Data = pd.read_csv ('raw.csv')
df_Data = df_Data.drop(['match_api_id','home_team_api_id','stage','away_team_api_id','FTR','FTResult'], axis=1) #'home_team_goal','away_team_goal'
df_Data = df_Data.drop(df_Data.columns[[0]],axis=1)
#df_Odds = df_Data[['B365H','B365D','B365A']]
#df_Data = df_Data.drop(['B365H','B365D','B365A'],axis=1)


y = []
for i in range(0,df_Data.shape[0]):
    home_goals = pd.to_numeric(df_Data.iloc[i].home_team_goal)
    away_goals = pd.to_numeric(df_Data.iloc[i].away_team_goal)
    win = home_goals-away_goals
    if win > 0:
        win = 1
    elif win < 0:
        win = -1
    else:
        win = -1
    y.append(win)

X = df_Data.drop(['home_team_goal','away_team_goal'],axis=1)
y = pd.DataFrame(y,columns=['Win'])
y = pd.to_numeric(y['Win'])

#split into train/test
train_num=250
valid_num=250
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42) #using df
#for train/val/test
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, train_size=train_num+valid_num, random_state=42) #using df
X_train2, X_valid, y_train2, y_valid = train_test_split(X, y, train_size=valid_num, random_state=42) #using df

df_Odds = X_test[['B365H','B365D','B365A']]
#X_train = X_train.drop(['B365H','B365D','B365A'],axis=1)
#X_test = X_test.drop(['B365H','B365D','B365A'],axis=1)
#standardize
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train),columns = X_train.columns,index=X_train.index)
X_train2 = pd.DataFrame(sc.fit_transform(X_train2),columns = X_train2.columns,index=X_train2.index)
X_valid = pd.DataFrame(sc.fit_transform(X_valid),columns = X_valid.columns,index=X_valid.index)
X_test = pd.DataFrame(sc.fit_transform(X_test),columns = X_test.columns,index=X_test.index)

###############PCA dim reduction (to play with), this uses its own preprocessing not the scaled data

n_reducedfeatures = 14

pca = decomposition.PCA(n_components=n_reducedfeatures)
#pca.fit(X_train)
#x_PCA = pca.transform(X_train)
x_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)
#X_test_PCA2 = pca.transform(X_test2)
print('PCA total variance sum: ' + str(sum(pca.explained_variance_ratio_)))
#print(pd.DataFrame(pca.components_,columns=X_train.columns,index = ['PC-1','PC-2','PC-3']))

#fig = px.scatter_3d(x=x_PCA[:,0],y=x_PCA[:,1],z=x_PCA[:,2],color = y_test.to_numpy().reshape(-1))
# number of components
n_pcs = pca.components_.shape[0]

# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = X_train.columns
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
print('Most Important Names: ' + str(most_important_names))

total_var = pca.explained_variance_ratio_.sum() * 100

labels = {str(i): f"PC {i+1}" for i in range(n_reducedfeatures)}
labels['color'] = 'Win'

fig = px.scatter_matrix(
    x_PCA,
    #dimensions=["Win", "Loss"],
    color=y_train,
    dimensions=range(n_reducedfeatures),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)
#fig.show()



##############logReg (to play with)
Pe_train = []   #Probability of error for training set
Pe_val = []     #Probability of error for validation set
Pe_test = []    #Probability of error for testing set
Cvalvec = np.geomspace(1e-3, 1e10, num=10)
for Cval in Cvalvec:
    clf = LogisticRegression(C=Cval,penalty='l2')
    clf.fit(X_train2,y_train2)
    #pred = logRegr.predict(X_test_PCA)
    #score = logRegr.score(X_test_PCA,y_test)
    #print('Logistic Regression score PCA: ' + str(score))
    Pe_train.append(1-clf.score(X_train2,y_train2))
    Pe_val.append(1-clf.score(X_valid,y_valid))
    Pe_test.append(1-clf.score(X_test2,y_test2))
    
Cval_opt = Cvalvec[np.argmin(Pe_val)]
print('----------LogRegression-----------')
print('Optimal value of C based on validation set: ' + str(Cval_opt))
print('Train error (LogRegression): ' + str(Pe_train[np.argmin(Pe_val)]))
print('Test error (LogRegression): ' + str(Pe_test[np.argmin(Pe_val)]))
print('----------------------------------')

'''
#RidgeRegression
clf = Ridge(alpha=100.0)
x_RR = clf.fit(x_PCA, y_train)
print('Ridge Regression score PCA: ' + str(x_RR.score(X_test_PCA,y_test)))
'''
##############logReg (to play with)
logRegr = LogisticRegression(C=Cvalvec[np.argmin(Pe_val)], penalty='l2')
logRegr.fit(X_train,y_train)
pred = logRegr.predict(X_test)
score = logRegr.score(X_test,y_test)
proba = logRegr.predict_proba(X_test)
print('Logistic Regression score: ' + str(score))
cnf_matrix = metrics.confusion_matrix(y_test, pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("bottom")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
#Text(0.5,257.44,'Predicted label');
wrong = []
df_prob_homewin = 1/df_Odds['B365H']
for i in range(0,len(df_prob_homewin)):
    if ((df_prob_homewin.iloc[[i][0]]>.5) and (y_test.iloc[i] == -1)):
        wrong.append(1)


        
from matplotlib.colors import ListedColormap


#plt.plot(np.linspace(0, 1, len(proba)),proba[:,1])
#plt.plot(np.linspace(0, 1, len(df_Odds_homewin)),df_Odds_homewin)
plt.scatter(np.linspace(0, 1, len(df_prob_homewin)),df_prob_homewin-proba[:,1], c=y_test)
plt.title("B365 vs. Logistic Regression Probability of Home Team Win")
plt.colorbar()
plt.xlabel("Match")
plt.ylabel("B365 - Logistic Regression Probability")
plt.show()


#RidgeRegression
clf = Ridge(alpha=100.0)
x_RR = clf.fit(X_train, y_train)
print('Ridge Regression score: ' + str(x_RR.score(X_test,y_test)))


#kernel rbf
Pe_train = []   #Probability of error for training set
Pe_val = []     #Probability of error for validation set
Pe_test = []    #Probability of error for testing set
numSvs = [] 
Gammavec = np.geomspace(1e-3, 1e10, num=10)
Cvalvec = np.geomspace(1e-3, 1e10, num=10)
'''
for Cval in Cvalvec:
    for Gammaval in Gammavec:
        print('Testing Gammaval = ' + str(Gammaval))
        #Train SVM w/ rbf kernel using C = 10 and Gammaval
        #You should compute the probability of error on all 3 sets and store them appropriately.
        #You should also track the number of support vectors.
        clf = svm.SVC(C=Cval,kernel='rbf', gamma=Gammaval)
        clf.fit(X_train2,y_train2)
        Pe_train.append(1-clf.score(X_train2,y_train2))
        Pe_val.append(1-clf.score(X_valid,y_valid))
        Pe_test.append(1-clf.score(X_test2,y_test2))
        numSvs.append(clf.support_vectors_.shape[0])
'''
for Gammaval in Gammavec:
    #print('Testing Gammaval = ' + str(Gammaval))
    #Train SVM w/ rbf kernel using C = 10 and Gammaval
    #You should compute the probability of error on all 3 sets and store them appropriately.
    #You should also track the number of support vectors.
    clf = svm.SVC(C=10,kernel='rbf', gamma=Gammaval)
    clf.fit(X_train2,y_train2)
    Pe_train.append(1-clf.score(X_train2,y_train2))
    Pe_val.append(1-clf.score(X_valid,y_valid))
    Pe_test.append(1-clf.score(X_test2,y_test2))
    numSvs.append(clf.support_vectors_.shape[0])
    
Gammaval_opt = Gammavec[np.argmin(Pe_val)]
print('----------RBF-----------')
print('Optimal value of Gamma based on validation set: ' + str(Gammaval_opt))
print('Train error (RBF): ' + str(Pe_train[np.argmin(Pe_val)]))
print('Test error (RBF): ' + str(Pe_test[np.argmin(Pe_val)]))
print('Number of support vectors: ' + str(numSvs[np.argmin(Pe_val)]))

print('Pe_test: ', Pe_test)
print('Pe_val: ', Pe_val)
print('numSvs: ', numSvs)
print('------------------------')

#y_predrbf = reg.predict(X_test)
#print(reg.score(X_test,y_test))

#Linear regression
reg =  LinearRegression().fit(X_train,y_train)
#lreg.fit(X_train,y_train)
print('Linear Regression score: ' +str(reg.score(X_test,y_test)))


