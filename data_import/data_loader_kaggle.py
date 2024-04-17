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



# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("database.sqlite")

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

print(df_Data.head())

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
        win = 0
    y.append(win)


X = df_Data.drop(['home_team_goal','away_team_goal'],axis=1)
y = pd.DataFrame(y)


df_Data.to_excel("encoded.xlsx")
df_Data.to_csv('encoded.csv')



#split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42) #using df

#standardize
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train),columns = X_train.columns,index=X_train.index)

sc = StandardScaler()
X_test = pd.DataFrame(sc.fit_transform(X_test),columns = X_test.columns,index=X_test.index)

###############PCA dim reduction (to play with), this uses its own preprocessing not the scaled data

n_reducedfeatures = 10

pca = decomposition.PCA(n_components=n_reducedfeatures)
pca.fit(X_train)
x_PCA = pca.transform(X_test)
print('PCA total variance sum: ' + str(sum(pca.explained_variance_ratio_)))
#print(pd.DataFrame(pca.components_,columns=X_train.columns,index = ['PC-1','PC-2','PC-3']))

fig = px.scatter_3d(x=x_PCA[:,0],y=x_PCA[:,1],z=x_PCA[:,2],color = y_test.to_numpy().reshape(-1))



##############logReg (to play with)
logRegr = LogisticRegression()
logRegr.fit(X_train,y_train)
pred = logRegr.predict(X_test)
score = logRegr.score(X_test,y_test)
print(score)




#kernel rbf
Cval = 150.0
epsilonval = 0.1
gammaval = 1.5

#reg = SVR(C=Cval, epsilon=epsilonval, kernel='rbf', gamma=gammaval)
#reg.fit(X.reshape(-1,1),ytrain)

#y_predrbf = reg.predict(xtest.reshape(-1,1))



fig.show()

