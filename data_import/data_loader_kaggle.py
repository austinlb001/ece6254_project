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

for i in range(0,df_Match.shape[0]):
    yearfind = pd.to_datetime(df_Match.iloc[i].date).year
    team_id = df_Match.iloc[i].home_team_api_id
    result = df_TeamAttributes[(df_TeamAttributes['team_api_id']==team_id) & (df_TeamAttributes['date'].dt.year==yearfind)]
    if not result.empty:
        tmp_attr.append(result)
        tmp_match.append(df_Match.iloc[[i]])
        #df_tmp = pd.concat([df_Match.iloc[i],result], axis=1)
        #df_Data = pd.concat([df_Data,df_tmprow], axis=0)
        #df_Data = pd.concat([df_Data,df_tmp], axis=0)
        #print(i)


tmp_attr = pd.concat(tmp_attr)
#tmp_attr = tmp_attr.drop(drop_list, axis=1)
tmp_attr = tmp_attr.reset_index(drop=True)
tmp_match = pd.concat(tmp_match)
tmp_match = tmp_match.reset_index(drop=True)
tmp_match = tmp_match[['home_team_goal', 'away_team_goal']]

df_Data = pd.concat([tmp_attr,tmp_match],axis=1)

drop_list = ['id', 'team_fifa_api_id', 'team_api_id', 'date', 'buildUpPlayDribbling'] #too many NA in dribbling

df_Data = df_Data.drop(drop_list, axis=1)
df_Data = df_Data.dropna()
    
y = []
for i in range(0,df_Data.shape[0]):
    home_goals = pd.to_numeric(df_Data.iloc[i].home_team_goal)
    away_goals = pd.to_numeric(df_Data.iloc[i].away_team_goal)
    win = home_goals-away_goals
    if win > 0:
        win = 1
    else:
        win = 0 #no draw
    y.append(win)
    

df_Data.to_excel("raw.xlsx")
df_Data.to_csv('raw.csv')

#encode strings
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = df_Data.select_dtypes(include=['object']).columns.tolist()
int_columns = df_Data.select_dtypes(include=['int']).columns.tolist()
df_encoded = df_Data[categorical_columns].apply(preprocessing.LabelEncoder().fit_transform)
data_encoded = pd.concat([df_Data[int_columns], df_encoded], axis=1)
df_Data = data_encoded

#standardize only the int colums
sc = StandardScaler()
df_Data[int_columns] = sc.fit_transform(df_Data[int_columns])
#X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)


df_Data.to_excel("encoded.xlsx")
df_Data.to_csv('encoded.csv')

X = df_Data
y = pd.DataFrame(y)

# Verify that result of SQL query is stored in the dataframe
print(df_Data.head())

con.close()





#split into train/test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42) #using np array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42) #using df



###############PCA dim reduction (to play with), this uses its own preprocessing not the scaled data

n_reducedfeatures = 2

pca = decomposition.PCA(n_components=n_reducedfeatures)
pca.fit(X_train)
x_PCA = pca.transform(X_test)

print(pd.DataFrame(pca.components_,columns=X_train.columns,index = ['PC-1','PC-2']))

graph = plt.scatter(x_PCA[:,0],x_PCA[:,1], c=y_test)
plt.colorbar(graph)
#plt.show()


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



plt.show()

