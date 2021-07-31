#!/usr/bin/env python
# coding: utf-8

# In[60]:


medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
from urllib.request import urlretrieve
urlretrieve(medical_charges_url , 'medical.csv')


# In[61]:


import pandas as pd 
medical_df = pd.read_csv('medical.csv')
medical_df


# In[62]:


medical_df.info() 
medical_df.describe() 


# In[63]:


#importing libraries reqd for data visualisation

import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[64]:


sns.set_style('darkgrid') 
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,6) 
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[65]:


#AGE ANALYSIS

medical_df.age.describe()

fig = px.histogram(medical_df, 
                   x = 'age', 
                   marginal = 'box', 
                   nbins = 47, 
                   title = 'Distribution of Age')
fig.update_layout(bargap = 0.1)
fig.show()


# In[13]:


# BODY MASS INDEX ANALYSIS

medical_df.bmi.describe()

fig = px.histogram(medical_df , 
                   x = 'bmi' , 
                   marginal = 'box', 
                   title = 'distribution of BMI', 
                   color_discrete_sequence = ['green'])
fig.update_layout(bargap = 0.1)
fig.show()


# In[ ]:





# In[66]:


#charges analysis , marking smokers and non smokers 

fig = px.histogram(medical_df, 
                   x='charges', 
                   marginal='box', 
                   color='smoker', 
                   color_discrete_sequence=['red', 'orange'], 
                   title='Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()


# In[67]:


medical_df.smoker.value_counts()


# In[68]:


fig = px.histogram(medical_df , x = 'smoker' , color = 'sex' , color_discrete_sequence=['red', 'green'] , title = 'Smoker Chart')
fig.show()


# In[25]:


# Ages vs Charges 

fig = px.scatter(medical_df, 
                 x='age', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='Age vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# In[69]:


# BMI vs charges plot 

fig = px.scatter(medical_df, 
                 x='bmi', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# In[70]:


# CORRELATION

medical_df.charges.corr(medical_df.age)


# In[71]:


medical_df.charges.corr(medical_df.bmi)


# In[72]:


smoker_values = {'no': 0, 'yes': 1}
smoker_numeric = medical_df.smoker.map(smoker_values)
medical_df.charges.corr(smoker_numeric)


# In[73]:


# Correlation matrix and heatmap

medical_df.corr()


# In[74]:


sns.heatmap(medical_df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix Heatmap');


# In[75]:


non_smoker_df = medical_df[medical_df.smoker == 'no']
non_smoker_df


# In[76]:


#visualize the relationship between "age" and "charges"

plt.title('Age vs. Charges')
sns.scatterplot(data = non_smoker_df, x = 'age', y = 'charges', alpha= 0.7, s = 15)

# Here if we observe the graph closely we can derive an approximate linear relation between age and charges 
# a straight line with an equation y = w*x + b 
# mathematically we call w as slope and b as intercept , but in ML they can be reffered as weight or parameters 


# In[77]:


# We'll try determine w and b for the line that best fits the data.
# This technique is called linear regression, and we call the above equation a linear regression model, because it models the relationship between "age" and "charges" as a straight line.
# The values in the "age" column of the dataset are called the inputs to the model and the values in the charges column are called "targets".
# We will now define a helper function named estimate_helper to help us find nearest correct value of the parameters (w , b)


# In[81]:


def estimate_charges(age , w , b):
    return w*age + b 


# In[84]:


# now we will find estimate charges for ranmdom values of w and b to reach to nearest correct values 

w = 50 
b = 100 

ages = non_smoker_df.age
print(ages)
estimated_charges = estimate_charges(ages , w , b)


# In[85]:


plt.plot(ages, estimated_charges, 'r-o');
plt.xlabel('Age');
plt.ylabel('Estimated Charges');


# In[86]:


# We can overlay this line on the actual data, so see how well our model fits the data.

target = non_smoker_df.charges

plt.plot(ages, estimated_charges , 'r', alpha = 0.9);
plt.scatter(ages, target, s = 8 , alpha = 0.8);
plt.xlabel('Age');
plt.ylabel('Charges')
plt.legend(['Estimate', 'Actual']);




def try_parameters(w,b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    
    estimated_charges = estimate_charges(ages, w, b)
    
    plt.plot(ages, estimated_charges, 'r', alpha = 0.9);
    plt.scatter(ages, target, s = 8 , alpha = 0.8);
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual']);




try_parameters(60,200)
try_parameters(400, 5000)

# RMSE : root mean squared error : gives us a measure of how varience of actual data and predicted data

import numpy as np 
 
def rmse(targets , predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))

w = 50
b = 100 

try_parameters(w, b) 

targets = non_smoker_df['charges'] 
predicted = estimate_charges(non_smoker_df.age , w , b) 

rmse(targets , predicted) # returns variance between actual and predicted data

from sklearn.linear_model import LinearRegression
model = LinearRegression() 
help(model.fit)

inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
print('inputs.shape :', inputs.shape)
print('targes.shape :', targets.shape)

model.fit(inputs, targets)
model.predict(np.array([[23],[37],[61]]))   

#getting predictions for all dataset non_smoker_df
predictions = model.predict(inputs)
predictions


    
