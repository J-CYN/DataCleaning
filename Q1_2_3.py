# %%
#
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %%
# Step 1:
# Can you tell if a person is unemployed by other features of their employment?
# Can you tell whether a person went to a college that is public or private based on other qualities in the dataset?

JobPl = pd.read_csv("Placement_Data_Full_Class.csv")
College = pd.read_csv("cc_institution_details.csv")
# %%
print(JobPl.head())
print(JobPl.isna().sum())
# %%
print(College.head())
print(College.isna().sum())
# %%
print(JobPl.info())
# %%
print(College.info())
# %%
print(JobPl.dtypes)
# %%
print(College.dtypes)
# %%
# Step 2:
# Write a generic question that this dataset could address.

# Can you tell if a person is employed by other features of their employment?
# Can you tell whether a person went to a college that is public or private based on other qualities in the dataset?

# What is a independent Business Metric for your problem? Think about the case study examples we have discussed in class.

# For the Jobs dataset, we could use this information to better target advertising perhaps for a company running a job training camp/trade school. Then the results from advertizing could be realized in sales metrics and registrations.
# For the College dataset, we could use this information to again target advertizing but perhaps to previous students of these groupings. As generally private university students may be more affluent giving us access to a group perhaps more receptive to luxury good sales, that we could track after advertizing to them.

# Drop unneeded variables
# %%
#Important, getting rid of empty columns or those that have many columns missing
Colleg_c = College.dropna(axis=1)
Colleg_c = Colleg_c.drop(["city","chronname","unitid","state","index","long_x","lat_y"], axis=1)

# %%
print(Colleg_c.head())
# %%
print(Colleg_c.info())


# Correct variable type/class as needed
#%%
colsJob = ["gender", "workex", "degree_t", "status", "hsc_b", "hsc_s","ssc_b","specialisation"]
JobPl[colsJob] = JobPl[colsJob].astype('category')

JobPl["salary"]=JobPl["salary"].fillna(0)

colsCollege = ["basic", "control", "level"]
Colleg_c[colsCollege] = Colleg_c[colsCollege].astype('category')

# Collapse factor levels as needed 
#%%
JobPl.status.value_counts()
#%%
Colleg_c.control.value_counts()

#%%
print(JobPl["ssc_b"].value_counts())

#%%
print(JobPl["specialisation"].value_counts())

#%%
#Reduce to 2 options: private or public
Colleg_c.control = (Colleg_c.control.apply(lambda x: x if x == "Public"
                               else "Private")).astype('category')

#%%
Colleg_c.level.value_counts()

# Apply one-hot encoding
# %%
cat_list_job=list(JobPl.select_dtypes('category'))
Jobs_encoded = pd.get_dummies(JobPl, columns=cat_list_job)
Jobs_encoded.info()

cat_list_college=list(Colleg_c.select_dtypes('category'))
Colleg_c_encoded = pd.get_dummies(Colleg_c, columns=cat_list_college)
Colleg_c_encoded.info()
#%%
print(Jobs_encoded.info())
#%%
print(Colleg_c_encoded.info())

# Normalize the continuous variables
#%%
numeric_cols_Job = list(Jobs_encoded.select_dtypes('number'))
Jobs_encoded[numeric_cols_Job] = MinMaxScaler().fit_transform(Jobs_encoded[numeric_cols_Job])

numeric_cols_College = list(Colleg_c_encoded.select_dtypes('number'))
Colleg_c_encoded[numeric_cols_College] = MinMaxScaler().fit_transform(Colleg_c_encoded[numeric_cols_College])

#%%
print(Jobs_encoded.info())
#%%
print(Colleg_c_encoded.info())

# Create target variable if needed
# Not needed for my example in particular

# Calculate the prevalence of the target variable 
# %%
# Calculate the prevalence (percentage of high-quality cereals)
prevalence_Jobs = np.sum(Jobs_encoded["status_Placed"]) / len(Jobs_encoded["status_Placed"])
print(prevalence_Jobs)
#%%
prevalence_College = np.sum(Colleg_c_encoded["control_Public"]) / len(Colleg_c_encoded["control_Public"])
print(prevalence_College)
#%%
print(f"Baseline/Prevalence(Jobs): {prevalence_Jobs:.2%}")
print(f"Baseline/Prevalence(College): {prevalence_College:.2%}")
# %%
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
#%%
print(train_Jobs["status_Placed"].value_counts())
print(tune_Jobs["status_Placed"].value_counts())
print(test_Jobs["status_Placed"].value_counts())


print(train_College["control_Public"].value_counts())
print(test_College["control_Public"].value_counts())
print(tune_College["control_Public"].value_counts())

# %%
# Step 3:

# What do your instincts tell you about the data. 
# Can it address your problem, what areas/items are you worried about? 

# My instincts tell me that while there are not an insignificant amount of cases, I would still want more data and more complete data.
# I also believe that the lack of a concrete data dictionary for the Jobs dataset hurt whatever we could try to achieve with it.
# As it means, apart from a few variables listed, much of the data is hard to understand or incomplete. There is also a lot of categorical data such as the column basic.
# These columns have many results that can not be fixed under feature reduction neatly.
# However, it should address the problem in the areas I'm worried about to a satisfiable degree, but for a longer term project perhaps a different dataset would be better suited.