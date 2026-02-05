# Step 4: Create functions for your two pipelines that produces the train and test datasets. The end result should be a series of functions that can be called to produce the train and test datasets for each of your two problems that includes all the data prep steps you took. This is essentially creating a DAG for your data prep steps. Imagine you will need to do this for multiple problems in the future so creating functions that can be reused is important. You don't need to create one full pipeline function that does everything but rather a series of smaller functions that can be called in sequence to produce the final datasets. Use your judgement on how to break up the functions.
#%%

import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data 

def readin():
    return pd.read_csv("Placement_Data_Full_Class.csv"), pd.read_csv("cc_institution_details.csv")

def Drop_Unneeded(College):
    Colleg_c = College.dropna(axis=1)
    Colleg_c = Colleg_c.drop(["city","chronname","unitid","state","index","long_x","lat_y"], axis=1)
    return Colleg_c

def correct_var_type(JobPl, Colleg_c):
    colsJob = ["gender", "workex", "degree_t", "status", "hsc_b", "hsc_s","ssc_b","specialisation"]
    JobPl[colsJob] = JobPl[colsJob].astype('category')

    JobPl["salary"]=JobPl["salary"].fillna(0)

    colsCollege = ["basic", "control", "level"]
    Colleg_c[colsCollege] = Colleg_c[colsCollege].astype('category')

    return JobPl, Colleg_c

def collapse_factors(Colleg_c):
    Colleg_c.control = (Colleg_c.control.apply(lambda x: x if x == "Public" else "Private")).astype('category')
    return Colleg_c

def OneHot(JobPl, Colleg_c):
    cat_list_job=list(JobPl.select_dtypes('category'))
    Jobs_encoded = pd.get_dummies(JobPl, columns=cat_list_job)

    cat_list_college=list(Colleg_c.select_dtypes('category'))
    Colleg_c_encoded = pd.get_dummies(Colleg_c, columns=cat_list_college)
    
    return Jobs_encoded, Colleg_c_encoded

def Normalize(Jobs_encoded, Colleg_c_encoded):
    numeric_cols_Job = list(Jobs_encoded.select_dtypes('number'))
    Jobs_encoded[numeric_cols_Job] = MinMaxScaler().fit_transform(Jobs_encoded[numeric_cols_Job])

    numeric_cols_College = list(Colleg_c_encoded.select_dtypes('number'))
    Colleg_c_encoded[numeric_cols_College] = MinMaxScaler().fit_transform(Colleg_c_encoded[numeric_cols_College])
    return Jobs_encoded, Colleg_c_encoded

def Partition(Jobs_encoded, Colleg_c_encoded):
    train_Jobs, test_Jobs = train_test_split(
        Jobs_encoded,
        train_size=55,
        stratify=Jobs_encoded["status_Placed"]
    )
    tune_Jobs, test_Jobs = train_test_split(
        test_Jobs,
        train_size=.5,
        stratify=test_Jobs["status_Placed"]
    )

    train_College, test_College = train_test_split(
        Colleg_c_encoded,
        train_size=55,
        stratify=Colleg_c_encoded["control_Public"]
    )
    tune_College, test_College = train_test_split(
        test_College,
        train_size=.5,
        stratify=test_College["control_Public"]
    )

    return train_Jobs, tune_Jobs, test_Jobs, train_College, tune_College, test_College

def total():
    JobPl, College = readin()
    Colleg_c=Drop_Unneeded(College)
    JobPl, Colleg_c = correct_var_type(JobPl, Colleg_c)
    Colleg_c = collapse_factors(Colleg_c)
    Jobs_encoded, Colleg_c_encoded = OneHot(JobPl, Colleg_c)
    Jobs_encoded, Colleg_c_encoded = Normalize(Jobs_encoded, Colleg_c_encoded)
    train_Jobs, tune_Jobs, test_Jobs, train_College, tune_College, test_College = Partition(Jobs_encoded, Colleg_c_encoded)
    return train_Jobs, tune_Jobs, test_Jobs, train_College, tune_College, test_College
#%%
# You can grab your test sets for both cases below with each of them split into Train, Tune, and Test.
train_Jobs, tune_Jobs, test_Jobs, train_College, tune_College, test_College = total()

