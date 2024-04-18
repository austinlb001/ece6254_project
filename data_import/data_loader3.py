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

import os

def FormatToDateFromDMY(dt):
    """
    Change format from Date Month Year to MM/DD/YYYY
    """
    datetime_obj = pd.to_datetime(dt, format="%d/%m/%y")
    if type(datetime_obj) != "NaTType":
        formatted_date = "{}/{}/{}".format(datetime_obj.date().month, datetime_obj.date().day, datetime_obj.date().year)
        # formatted_date = datetime_obj.strftime("%m/%d/%y")
    else:
        formatted_date = None
    
    return formatted_date


def FormatToDateFromDT(dt):
    """
    Change format from DateTime to MM/DD/YYYY
    """
    datetime_obj = pd.to_datetime(dt)
    if type(datetime_obj) != "NaTType":
        formatted_date = "{}/{}/{}".format(datetime_obj.date().month, datetime_obj.date().day, datetime_obj.date().year)
    else:
        formatted_date = None
    # formatted_date = datetime_obj.date().strftime("%m/%d/%Y")
    return formatted_date
    

def FormatDate(dt, type : int):
    """
    Format date to MM/DD/YYYY from different formats
    Params:
    - dt : Date time. Can be str or datetime.
    - type : 1 - datetime, 2 - DMY
    """
    if type == 1:
        return FormatToDateFromDT(dt)
    elif type == 2:
        return FormatToDateFromDMY(dt)

    return None

# Read sqlite into a Pandas dataframe.
con = sqlite3.connect("database.sqlite")

start_year = 2010 #we have no team attribute data before 2010

df_Teams = pd.read_sql_query("SELECT * from Team", con)
df_Match = pd.read_sql_query("SELECT * from Match", con)

##### Collating and formatting Kaggle Data #####

# Merge the matches with team to collate match data with home and away teams.
merged_drop_cols = ["id_y", "team_api_id", "team_fifa_api_id", "team_short_name"] # These are the columns from the teams table that are redundant.
temp_merged_df = df_Match.merge(df_Teams, left_on=['home_team_api_id'], right_on=['team_api_id'], how='inner')
temp_merged_df.rename(columns={"team_long_name":"home_team", "id_x":"id"}, inplace=True)
temp_merged_df.drop(labels=merged_drop_cols, axis=1, inplace=True)
df_MatchAndTeam = temp_merged_df.merge(df_Teams, left_on=["away_team_api_id"], right_on=["team_api_id"], how="inner")
df_MatchAndTeam.rename(columns={"team_long_name":"away_team", "id_x":"id"}, inplace=True)
df_MatchAndTeam.drop(merged_drop_cols, axis=1, inplace=True)

# Format date to MM/DD/YYYY
df_MatchAndTeam["Date"] = df_MatchAndTeam["date"].apply(lambda x: FormatDate(x, 1))
df_MatchAndTeam.drop(labels=["date"], axis=1, inplace=True)

# Format team names
df_MatchAndTeam["HomeTeam"] = df_MatchAndTeam["home_team"].str.lower().str.replace(" ","")
df_MatchAndTeam["AwayTeam"] = df_MatchAndTeam["away_team"].str.lower().str.replace(" ","")
df_MatchAndTeam.drop(labels=["home_team", "away_team"], axis=1, inplace=True)

##### End of formatting kaggle data #####

##### Collating and Formatting Football Co Data #####
folder_path = 'C:\\Users\\kandi\\OneDrive\\Documents\\Georgia Tech\\ECE 6254 Stat Machine Learning\\Project\\ece6254_project\\data_import\\FootballCoData\\'
file_list = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]

temp_dfs_list = []
for file in file_list:
    df = pd.read_csv(file)
    temp_dfs_list.append(df)

df_MatchAndOdds = pd.concat(temp_dfs_list, axis=0, ignore_index=True)

# Drop all the odds except for B365
cols_to_drop = ["BWH","BWD","BWA","GBH","GBD", "GBA","IWH","IWD",
                "IWA","LBH","LBD","LBA","SBH","SBD","SBA", "WHH",
                "WHD","WHA","SJH","SJD","SJA","VCH","VCD","VCA",
                "BSH","BSD","BSA","Bb1X2","BbMxH","BbAvH","BbMxD",
                "BbAvD","BbMxA","BbAvA","BbOU","BbMx>2.5","BbAv>2.5",
                "BbMx<2.5","BbAv<2.5","BbAH","BbAHh","BbMxAHH","BbAvAHH",
                "BbMxAHA","BbAvAHA","PSH","PSD","PSA","PSCH","PSCD","PSCA"]

df_MatchAndOdds.drop(cols_to_drop, axis=1, inplace=True)

#Format the date time
df_MatchAndOdds["Date"] = df_MatchAndOdds["Date"].apply(lambda x: FormatDate(x, 2))

# Format Teams to Lower Case
df_MatchAndOdds["HomeTeam"] = df_MatchAndOdds["HomeTeam"].str.lower().str.replace(" ","")
df_MatchAndOdds["AwayTeam"] = df_MatchAndOdds["AwayTeam"].str.lower().str.replace(" ","")


# df_MatchAndOdds.sample(25).to_csv("./combined_data.csv", index=False)
# df_MatchAndTeam.sample(25).to_csv("./combined_data2.csv", index=False)


df_CombinedMatchOdds = df_MatchAndTeam.merge(df_MatchAndOdds, left_on=["Date", "HomeTeam", "AwayTeam"], right_on=["Date", "HomeTeam", "AwayTeam"])
df_CombinedMatchOdds.to_csv("./CombinedMatchOdds.csv", index=False)

