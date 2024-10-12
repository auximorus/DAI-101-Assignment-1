#!/usr/bin/env python
# coding: utf-8

# In[566]:


import math
import statistics
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from scipy.stats import chi2_contingency
from scipy import stats as st
import os


# In[567]:


df = pd.read_csv("../dataset/pokedex.csv")


# In[568]:


print(df.info())
print(df.head(10))
numericalvars = ['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed', 'Total']


# In[569]:


def univariate_numerical(data, col):
    print("Column Name:", col)
    
    # Displaying summary statistics
    print(data[col].describe())
    
    # Set up the figure with 3 subplots: Histogram, Density Plot, and Box Plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram
    ax[0].hist(data[col], bins=min(20, int(data[col].max() / 5)), edgecolor="blue")
    ax[0].set_title(f'Histogram of {col}')
    ax[0].set_xlabel(col)
    ax[0].set_ylabel('Frequency')
    
    # Density Plot (KDE)
    sns.kdeplot(data[col], ax=ax[1], color='green')
    ax[1].set_title(f'Density Curve of {col}')
    ax[1].set_xlabel(col)
    ax[1].set_ylabel('Density')
    
    # Box Plot
    sns.boxplot(data[col], ax=ax[2], color='orange')
    ax[2].set_title(f'Box Plot of {col}')
    ax[2].set_xlabel(col)
    
    plt.tight_layout()
    plt.show()


# In[570]:


print("univariate Analysis of Numerical variables")
univariate_numerical(df,numericalvars[0])


# In[571]:


univariate_numerical(df,numericalvars[1])


# In[572]:


univariate_numerical(df,numericalvars[2])


# In[573]:


univariate_numerical(df,numericalvars[3])


# In[574]:


univariate_numerical(df,numericalvars[4])


# In[575]:


univariate_numerical(df,numericalvars[5])


# In[576]:


univariate_numerical(df,numericalvars[6])


# In[577]:


def univariate_categorical(data, col):
    print(f"Univariate Analysis for Categorical Variable: {col}")
    print(data[col].value_counts())
    
    # Bar Plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col, data=data, palette='Set2', order=data[col].value_counts().index)
    plt.title(f'Bar Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)  
    plt.show()

# print(df["Type 1"].describe())
# sns.countplot(df["Type 1"],color = "red")
# plt.show()


# In[578]:


univariate_categorical(df,"Type 1")


# In[579]:


univariate_categorical(df,"Type 2")


# In[580]:


print(df.columns)


# In[581]:


cat = []        # Categorical columns
num = []        # Numerical columns
identifier = [] # Columns to exclude from cat/num

# Columns to exclude from categorical or numerical classification
exclude = ['Image', 'Index', 'Name']

# Loop through each column in the dataset
for column in df.columns:
    if column in exclude:
        identifier.append(column)  # Add to identifier list if it's in the exclude list
    elif pd.api.types.is_numeric_dtype(df[column]):
        if df[column].nunique() > 10:
            num.append(column)  # Add to numerical list if it has more than 10 unique values
        else:
            cat.append(column)  # Otherwise, treat it as categorical
    else:
        cat.append(column)  # Non-numeric columns are treated as categorical

# Print out the categorized lists
print("Categorical columns:", cat)
print("Numerical columns:", num)
print("Identifier columns:", identifier)


# In[582]:


# BIVARIATE ANALYSIS

# numerical numerical

sns.scatterplot(data = df,x="Total",y="HP")


# In[583]:


# categorical numerical
sns.barplot(data=df,x="Type 1",y="Defense")
plt.xticks(rotation=90)
plt.show()


# In[584]:


sns.boxplot(data=df, x="Type 1", y="Defense")
plt.xticks(rotation=90)
plt.show()


# In[585]:


sns.violinplot(x='Type 1',y='Attack',data=df ,palette='rainbow')
plt.xticks(rotation=90)
plt.show()


# In[586]:


cat


# In[587]:


# categorical categorical

pd.crosstab(df['Type 1'],df['Type 2'])


# In[588]:


sns.heatmap(pd.crosstab(df['Type 1'],df['Type 2']))


# In[589]:


num


# In[590]:


# Option 1: Z-Score

z = np.abs(scipy.stats.zscore(df["Attack"]))
print(z)

#dropping outliers
df = df[(z < 3)]


# In[591]:


# chi 2 test caegorical and categorical

contigency= pd.crosstab(df['Attack'], df['Defense']) 
contigency


# In[592]:


c, p, dof, expected = chi2_contingency(contigency) 
print(p)


# In[593]:


# Identify non-numeric values
non_numeric_values = df[~df['Type 1'].str.isdigit()]
print(non_numeric_values)

# Convert non-numeric values to NaN
df['Type 1'] = pd.to_numeric(df['Type 1'], errors='coerce')

# Optionally, drop rows with NaN values
df = df.dropna(subset=['Type 1'])

df['Type 1'] = df['Type 1'].astype(int)


# In[594]:


cat


# In[595]:


num


# In[596]:


# t-test - Categorical and Numerical
df = pd.read_csv("../dataset/pokedex.csv")
tr=df.dropna(axis=0,how='any')
tr.describe()

a = tr[tr['Total'] >450]['Speed']
b = tr[tr['Total'] <=450 ]['Speed']
print(a,b)
t,p =st.ttest_ind(a,b)
print(f't-statistic: {t}, p-value: {p}')


# In[597]:


# ANOVA - Categorical and Numerical
fvalue, pvalue = st.f_oneway(a,b)
print(fvalue, pvalue)


# In[598]:


# Linear - Numerical and Numerical
x = tr['Attack'].to_numpy()
y = tr['Defense'].to_numpy()
np.corrcoef(x, y)


# In[599]:


#Z-score normalization
from sklearn.preprocessing import StandardScaler

# Drop missing values for simplicity
df.dropna(subset=['Defense'], inplace=True)

# Z-score normalization of the 'PAS' column
scaler = StandardScaler()
df['Defense_zscore'] = scaler.fit_transform(df[['Defense']])

# Display the first few rows
print(df[['Defense', 'Defense_zscore']].head())


# In[600]:


# decimal scaling normalization

# Find the maximum absolute value in the PAS column
max_val = np.max(np.abs(df['Defense']))

# Calculate j (the smallest integer such that max(|x'|) < 1)
j = np.ceil(np.log10(max_val))

# Apply decimal scaling normalization
df['Defense_scaled'] = df['Defense'] / (10**j)

# Display the first few rows
print(df[['Defense', 'Defense_scaled']].head())


# In[601]:


# Check for infinite values
print(df[['HP', 'Attack', 'Defense', 'Total']].isin([np.inf, -np.inf]).sum())

# Check for extremely large values
print(df[['HP', 'Attack', 'Defense', 'Total']].max())

# Replace infinite values with NaN
df[['HP', 'Attack', 'Defense', 'Total']] = df[['HP', 'Attack', 'Defense', 'Total']].replace([np.inf, -np.inf], np.nan)

# Optionally, drop rows with NaN values
df.dropna(subset=['HP', 'Attack', 'Defense', 'Total'], inplace=True)


# In[602]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Normalize the variables
scaler = MinMaxScaler()
df[['HP', 'Attack', 'Defense', 'Total']] = scaler.fit_transform(df[['HP', 'Attack', 'Defense', 'Total']])

# Check the result
df.head()


# In[603]:


# Encode the Speed variable
le = LabelEncoder()
df['Speed'] = le.fit_transform(df['Speed'])

# Check the result
df.head()


# In[604]:


# Aggregate the data by Positioning and get summary statistics
agg_df = df.groupby(['Speed']).agg({'HP': 'mean', 
                                      'Attack': 'mean', 
                                      'Defense': 'mean', 
                                      'Total': 'mean'})

# Check the result
print(agg_df)


# In[605]:


# Log-transform the OVR and PAC
df['HP'] = np.log(df['Attack'])
df['Attack'] = np.log(df['HP'])

# Check the result
print(df.head(20))


# In[606]:


# Replace infinite values with NaN
df[['HP', 'Attack', 'Defense', 'Total']] = df[['HP', 'Attack', 'Defense', 'Total']].replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN values
df.dropna(subset=['HP', 'Attack', 'Defense', 'Total'], inplace=True)

# # Verify the DataFrame state before scaling
# print("DataFrame before scaling:")
# print(df[['HP', 'Attack', 'Defense', 'Total']].describe())

# Scale the variables
scaler = StandardScaler()
df[['HP', 'Attack', 'Defense', 'Total']] = scaler.fit_transform(df[['HP', 'Attack', 'Defense', 'Total']])

# Check the result
print("DataFrame after scaling:")
print(df.head())


# In[607]:


# Select only numerical columns for the PairGrid
numerical_columns = ['HP', 'Attack', 'Defense', 'Total']  # Adjust this list as necessary

# Drop rows with NaN values in the specified columns
df = df.dropna(subset=['Total', 'HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed'])

# Create a filtered DataFrame to check for constant values
filtered_df = df[['Total', 'HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']].copy()

# Filter out rows where all values are constant
filtered_df = filtered_df.loc[(filtered_df.nunique(axis=1) > 1)]

# Create the PairGrid using the original DataFrame (or filtered if you prefer)
g = sns.PairGrid(df[numerical_columns])
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot, colors="C0")
g.map_diag(sns.kdeplot, lw=2)
plt.show()


# In[ ]:




