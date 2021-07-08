#!/usr/bin/env python
# coding: utf-8

# # PROJECT: Cervical Cancer Prediction Model

# # Task: Import the important libraries 

# In[1]:


# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().system('pip install plotly')
import plotly.express as px


# In[2]:


# Importing the dataset
cer_cancer= pd.read_csv("cervical_cancer.csv")


# # Task: Import the dataset

# In[3]:


# Exploreing the data
cer_cancer


# In[4]:


# Input Variables

# (int) Age
# (int) Number of sexual partners
#  (int) First sexual intercourse (age)
# (int) Num of pregnancies
# (bool) Smokes
# (bool) Smokes (years)
# (bool) Smokes (packs/year)
# (bool) Hormonal Contraceptives
# (int) Hormonal Contraceptives (years)
# (bool) IUD ("IUD" stands for "intrauterine device" and used for birth control
# (int) IUD (years)
# (bool) STDs (Sexually transmitted disease)
# (int) STDs (number)
# (bool) STDs:condylomatosis
# (bool) STDs:cervical condylomatosis
# (bool) STDs:vaginal condylomatosis
# (bool) STDs:vulvo-perineal condylomatosis
# (bool) STDs:syphilis
# (bool) STDs:pelvic inflammatory disease
# (bool) STDs:genital herpes
# (bool) STDs:molluscum contagiosum
# (bool) STDs:AIDS
# (bool) STDs:HIV
# (bool) STDs:Hepatitis B
# (bool) STDs:HPV
# (int) STDs: Number of diagnosis
# (int) STDs: Time since first diagnosis
# (int) STDs: Time since last diagnosis
# (bool) Dx:Cancer
# (bool) Dx:CIN
# (bool) Dx:HPV
# (bool) Dx

#Target Varibles
# These are the four most common test for cervical cancer diagnosis
# (bool) Hinselmann 
# (bool) Schiller
# (bool) Citology
# (bool) Biopsy


# # Task: Perform Explanatory Data Analysis

# In[5]:


# Getting information about the DataFrame
cer_cancer.info()


# In[6]:


# The statistics of the data frame
cer_cancer.describe()


# In[7]:


#Exploring the data
cer_cancer


# In[8]:


# Notice the '?' which indicates missing data
# The above dataset has '?' where data was not disclosed by the patient
# We replace '?' with NaN 
cer_cancer= cer_cancer.replace('?', np.nan)
cer_cancer


# In[10]:


# Plot heatmap
cer_cancer.isnull()


# In[12]:


# Plot the heatmap to identify null valuses
plt.figure(figsize = (20,20))
sns.heatmap(cer_cancer.isnull(), yticklabels = False)


# In[13]:


# we notice a lot of data missing from two particular columns in the heatmap


# In[14]:


# Data frame information
cer_cancer.info()


# In[15]:


# Since STDs: Time since first diagnosis  and STDs: Time since last diagnosis have more than 80% missing values 
# we are dropping them 
cer_cancer= cer_cancer.drop( columns = ['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])
cer_cancer


# In[16]:


# Since most of the column type is object, we are not able to get the statistics of the dataframe
# We convert the object column type to  numeric type
cer_cancer= cer_cancer.apply(pd.to_numeric)
cer_cancer.info()


# In[17]:


# The statistics of the dataframe
cer_cancer.describe()


# In[18]:


# The mean of all the varibles
cer_cancer.mean()


# In[20]:


# Replace null values with mean
cer_cancer= cer_cancer.fillna(cer_cancer.mean())
cer_cancer


# In[21]:


# Plot of the Nan heatmap
sns.heatmap(cer_cancer.isnull(), yticklabels= False)


# In[22]:


# The above heatmap shows that there are no null values ( one homogeneous colour is seen) which is exactly what we are looking for


# # TASK: PERFORM DATA VISUALIZATION

# In[23]:


# The correlation matrix
corr_mat= cer_cancer.corr()
corr_mat


# In[24]:


# Plot of the correlation matrix
plt.figure(figsize = (30, 30))
sns.heatmap(corr_mat, annot = True)
plt.show()


# In[26]:


# We plot the histogram for the entire DataFrame
cer_cancer.hist(bins= 10, figsize = (30, 30), color = 'b')


# # TASK: PREPARE THE DATA BEFORE TRAINING

# In[27]:


cer_cancer


# In[28]:


# We set Biopsy as our target variable

target_df= cer_cancer['Biopsy']
input_df= cer_cancer.drop(columns = ['Biopsy'])

# Shape of target_df
target_df.shape


# In[29]:


# Shape of input_df
input_df.shape


# In[30]:


# Convert my input_df and target_df into float32
# input is X
# output is y
X = np.array(input_df).astype('float32')
Y = np.array(target_df).astype('float32')


# In[31]:


# reshaping the array from (421570,) to (421570, 1)
# Y = Y.reshape(-1,1)
Y.shape


# In[32]:


# Normalisation of the data
# scaling the data before feeding the model

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[33]:


# We checkout X
X


# In[34]:


# We get a normalised dataset


# In[35]:


# When we train a model we need to take the entire dataset and divide it into a training, testing and cross-validation data
# Training data is fed to the model so that the model could learn
# cross-validation data is used to make sure the model is generating good results and not overfitting the data

# spliting the data in to test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=0.2)
# we split the data with 80% as training data and 20% as testing data

# we then slpit testing data into validation data and testing data
X_test, X_val, Y_test, Y_val= train_test_split(X_test, Y_test, test_size= 0.5)


# # TASK: TRAIN AND EVALUATE XGBOOST CLASSIFIER

# In[36]:


# Install XGBOOST

get_ipython().system('pip install xgboost')


# In[37]:


# Train an XGBoost classifier model 
import xgboost as xgb

#Specify the learning rate
#Specify the depth of the tree
#Specify the number to estimators/model to be used

model= xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators= 10)

# Train the model
model.fit(X_train, Y_train)


# In[38]:


# Evaluate the models performance
result_train= model.score(X_train, Y_train)
result_train


# In[39]:


# we see that we have achieved 97% accuracy with our training data


# In[40]:


# We predict the score of the trained model using the testing dataset
result_test= model.score(X_test, Y_test)
result_test


# In[41]:


# we see that we have achieved 94% accuracy with our testing data


# In[42]:


# Next we predict the score of the trained model using the testing dataset

Y_predict= model.predict(X_test)


# In[43]:


# Then we print the classification report

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(Y_test,Y_predict))


# In[44]:


# we observe precision of 99% on class zero which is pretty good, however the precision and recall for class1 is not that good


# In[45]:


# We print the confusion matrix

con_matrix= confusion_matrix(Y_test,Y_predict)
sns.heatmap(con_matrix, annot= True)


# In[ ]:


# The model corretly classify 74(top left) and 7(bottom right) samples and misclassify 4(top right) and 1(bottom left) samples as seen above in the heatmap


# # THANK YOU!!
