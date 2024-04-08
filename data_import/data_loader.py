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


year_range = range(2010,2023)

data = list()

#load all year ranges selected
for i, year in enumerate(year_range):
    url_string = 'https://www.football-data.co.uk/mmz4281/%.2i%.2i/E0.csv' %(abs(year)%100,abs(year+1)%100)
    #url_string = 'https://sports-statistics.com/database/soccer-data/england-premier-league-%.2i-to-%.2i.csv' %(year,year+1)
    data.append(pd.read_csv(url_string))

#select which features we want and aggreagate data and create np array from df
    
#feature_list = ['FTR','HomeTeam','AwayTeam','HTHG','HTAG','HS']
#feature_list = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','Referee','HS','AS','HST',
 #'AST','HF','AF','HC','AC','HY','AY','HR','AR','B365H','B365D','B365A','BWH','BWD','BWA',
 #'IWH','IWD','IWA','LBH','LBD','LBA','PSH','PSD','PSA','WHH','WHD','WHA','VCH','VCD','VCA',
 #'Bb1X2','BbMxH','BbAvH','BbMxD','BbAvD','BbMxA','BbAvA','BbOU','BbMx>2.5','BbAv>2.5','BbMx<2.5',
 #'BbAv<2.5','BbAH','BbAHh','BbMxAHH','BbAvAHH','BbMxAHA','BbAvAHA','PSCH','PSCD','PSCA']
#feature_list = ['FTR','HomeTeam','AwayTeam','FTHG','FTAG','HTHG','HTAG','HTR','HS','AS','HST',
 #'AST','HF','AF','HC','AC','HY','AY','HR','AR']
feature_list = ['FTR','FTHG','FTAG','HTHG','HTAG','HTR','HS','AS','HST',
 'AST','HF','AF','HC','AC','HY','AY','HR','AR']

#initial test then looped below, can clean when have time
data[0] = data[0].dropna()
encoder = OneHotEncoder(sparse_output=False)
data[0] = data[0].filter(items=feature_list)
categorical_columns = data[0].select_dtypes(include=['object']).columns.tolist()
one_hot_encoded = encoder.fit_transform(data[0][categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
data_encoded = pd.concat([data[0], one_hot_df], axis=1)
data_encoded = data_encoded.dropna()
clean_y = data_encoded.FTR.to_numpy()
data_encoded = data_encoded.drop(categorical_columns, axis=1)
data_encoded = data_encoded.drop(['FTR_A','FTR_D','FTR_H'], axis=1)
clean_data = data_encoded.to_numpy()
clean_data_df = data_encoded



for nyear in range(1,len(data)):
    data[nyear] = data[nyear].dropna()
    data[nyear] = data[nyear].filter(items=feature_list)
    #categorical_columns = data[nyear].select_dtypes(include=['object']).columns.tolist()
    one_hot_encoded = encoder.fit_transform(data[nyear][categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    data_encoded = pd.concat([data[nyear], one_hot_df], axis=1)
    data_encoded = data_encoded.dropna()
    clean_y = np.append(clean_y,data_encoded.FTR.to_numpy())
    data_encoded = data_encoded.drop(categorical_columns, axis=1)
    data_encoded = data_encoded.drop(['FTR_A','FTR_D','FTR_H'], axis=1) #drop the FTR classifier
    clean_data = np.vstack((clean_data,data_encoded.to_numpy()))
    clean_data_df = pd.concat([clean_data_df, data_encoded], ignore_index=True, axis=0)
    

#clean_y_num = []
#for i in range(0,len(clean_y)): #i am sure there is built in function or something better, just hacking this now
    #if clean_y[i] == 'H':
        #clean_y_num[i] = 1
    #elif clean_y[i] == 'A':
        #clean_y_num[i] = 0
    #else:
        #clean_y_num[i] = 2


    
X = clean_data #np array
X_df = clean_data_df #df
y = clean_y

#standardize using df
sc = StandardScaler()
X_df = pd.DataFrame(sc.fit_transform(X_df), columns=X_df.columns)
#X = sc.transform(X)



#split into train/test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42) #using np array
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=.2, random_state=42) #using df



###############PCA dim reduction (to play with), this uses its own preprocessing not the scaled data

n_reducedfeatures = 2

pca = decomposition.PCA(n_components=n_reducedfeatures)
pca.fit(X_train)
x_PCA = pca.transform(X_test)

print(pd.DataFrame(pca.components_,columns=X_train.columns,index = ['PC-1','PC-2']))


y_test_num = []
for i in range(0,len(y_test)): #i am sure there is built in function or something better, just hacking this now to see how plots look
    if y_test[i] == 'H':
        y_test_num.append(1)
    elif y_test[i] == 'A':
        y_test_num.append(-1)
    else:
        y_test_num.append(0)

graph = plt.scatter(x_PCA[:,0],x_PCA[:,1], c=y_test_num)
plt.colorbar(graph)

##############logReg (to play with)
logRegr = LogisticRegression()
logRegr.fit(X_train,y_train)
pred = logRegr.predict(X_test)
#score = logRegr.score(pred,y_test)
#print(score)

#how many were wrong?
fake_score = 0
for i,predval in enumerate(pred):
    if predval!=y_test[i]:
        fake_score = fake_score+1 #+1 for each wrong
print(fake_score/len(pred))



#kernel rbf
Cval = 150.0
epsilonval = 0.1
gammaval = 1.5

#reg = SVR(C=Cval, epsilon=epsilonval, kernel='rbf', gamma=gammaval)
#reg.fit(X.reshape(-1,1),ytrain)

#y_predrbf = reg.predict(xtest.reshape(-1,1))



plt.show()



