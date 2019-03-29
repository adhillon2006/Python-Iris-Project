# general libaries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# ml libraries
import scipy.interpolate
import scipy as sp
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# set directory
os.chdir("C:/Users/adhillon/OneDrive - NVIDIA Corporation/AnacondaProjects/OP 802 Programming/Project/")

# load iris_not_clean.data
iris_data = pd.read_table("iris_not_clean.data",sep="|",
                          names=['sepal length', 'sepal width', 'petal length','petal width','type'],
                          skiprows=[0,1])

# How many rows and columns are in the file
iris_data.shape

# count number of na in the file
"There are {} NAs in the file".format(iris_data.isna().sum().sum())

# Which rows contains nans and how many nans in each of these rows
null_columns=iris_data.columns[iris_data.isna().any()]
print(iris_data[iris_data.isna().any(axis=1)][null_columns]) # prints rows contains nans
iris_data[iris_data.isna().any(axis=1)][null_columns].index # gets rows contains nans
iris_data[iris_data.isnull().any(axis=1)][null_columns].isna().sum(axis=1) # nans in each rows

# How many nans per columns?
iris_data.isna().sum()

# The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa" based on the original paper.
# Is that what you have? If not, fix it
iris_data.iloc[36] # petal width is 0.1
iris_data.iloc[36,3]=0.2 # change petal width to 0.2
print(iris_data.iloc[36])

# The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa", where the errors are in the second and third features. Fix it
iris_data.iloc[39]
iris_data.iloc[39,[1,2]] #  sepal width 3.1 | petal length  1.5
iris_data.iloc[39,[1,2]] = [3.6,1.4]

# What is the mean and variance
# for each column and use the function describe to get insights about the data
iris_mean = np.mean(iris_data)
iris_var = np.var(iris_data)
iris_describe = iris_data.describe()

# drop na values
iris_data_clean = iris_data.dropna()
np.mean(iris_data_clean)
np.var(iris_data_clean)

# change type to float to get correlation
iris_data_clean['sepal length'] = iris_data_clean['sepal length'].astype(float)
iris_data_clean['sepal width'] = iris_data_clean['sepal width'].astype(float)
iris_data_clean['petal length'] = iris_data_clean['petal length'].astype(float)
iris_data_clean['petal width'] = iris_data_clean['petal width'].astype(float)

# What is the correlation between the fourth column
# and each of the other three columns individually? Any observations
for data in iris_data_clean.columns:
    try:
        print(np.correlate(iris_data_clean['petal width'],iris_data_clean[data]))
    except TypeError:
        pass

# How many records exist for each class. Hint, you can do this in one line by using groupby
iris_data_clean.groupby('type').count()

np.random.seed(1234) # random seed to 1234
sample = np.random.randint(0,len(iris_data_clean.index),(20)) # this is a tuple

# tuple converted to list
list1 = []
for x in sample:
    list1.append(x)

# randomly pick 20 samples of the data and display it
iris_data_clean.iloc[list1]

# Plot histogram for all the data attributes
i=0
fig = plt.figure(figsize=(15,10))
for plots_names in iris_data_clean.columns[0:4]:
    i+=1
    fig.add_subplot(2, 2, i).title.set_text(plots_names)
    fig.add_subplot(2, 2, i).hist(iris_data_clean[plots_names])
fig

# Plot histogram for all the data attributes per feature, i.e. grouped by features
i=0
fig = plt.figure(figsize=(15,10))
for features in iris_data_clean['type'].unique():
    iris_filtered=iris_data_clean[iris_data_clean.type==features]
    for plots_names in iris_filtered.columns[0:4]:
        i+=1
        fig.add_subplot(3, 4, i).title.set_text(features+" "+plots_names)
        fig.add_subplot(3, 4, i).hist(iris_filtered[plots_names])
fig





# Part 3: Statistical Analysis
# In this part, you will explore some curve fitting and dimensionality reductions attributes

# Use Scipy pdf fitting to do a curve fitting for the petal-length
# Plot the normalized histogrm of the petal-length and the estimated pdf on the same figure
# Generate new 500 samples using the estimated pdf for the petal-length
# Calculate the mean of the new samples and compare how close it is to the mean of the given data

xl=iris_data_clean['petal length'] # information to fit
xs1 = np.linspace(xl.min(), xl.max(), 500)
kde1 = sp.stats.gaussian_kde(xl)

fig = plt.figure(figsize=(8, 6))

plt.hist(xl, density=True, label='Normalized Histogram')
plt.plot(xs1, kde1(xs1), 'k-', label="Estimated Density")
plt.xlabel('Samples')
plt.ylabel('Density')
plt.legend()

scipy.stats.ttest_1samp(kde1.resample(2000)[0],xl.mean())

# Use Scikit to do PCA on the IRIS dataset
# do a bar plot that shows the importance of info in each of the new dimensions
# use PCA to reduce the number of attributes by 1. Note that for all the next parts, you will use the reduced data

# Create a regular PCA model
pca_all = PCA()
reduced_data_pca = pca_all.fit_transform(iris_data_clean.iloc[0:,[1,2,3]]) # can't use features or error
pca_all.explained_variance_ratio_
plt.bar(np.arange(3),pca_all.explained_variance_ratio_)
reduced_data_pca.shape

pca = PCA(n_components=3)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(iris_data_clean.iloc[0:,[1,2,3]])

# Inspect the shape
reduced_data_pca.shape
plt.bar(np.arange(3),pca.explained_variance_ratio_)


# Part 4: Machine Learning
# In this part, you will explore the use of supervised and non supervised learning

# Non-Supervised Learning
# using Kmeans, divide the data into different clusters. The number of clusters should be the same as the number of categories you have in the data
# Do scatter plot for each two combination of the three dimensions together (0 vs 1), (0 vs 2), (1 vs 2). Use the kmeans labels to color the points in the scatter plots
plt.scatter(reduced_data_pca[:,0],reduced_data_pca[:,1])
plt.scatter(reduced_data_pca[:,0],reduced_data_pca[:,2])
plt.scatter(reduced_data_pca[:,1],reduced_data_pca[:,2])

kmeans_model=KMeans(n_clusters=2) # build k means model
kmeans_model.fit(reduced_data_pca) # fit model
kmeans_model.cluster_centers_ # printing out the centeroid

kmeans_model.labels_ # get cluster labels_
plt.scatter(reduced_data_pca[:,0],reduced_data_pca[:,1], c=kmeans_model.labels_, cmap='rainbow' )
plt.scatter(reduced_data_pca[:,0],reduced_data_pca[:,2], c=kmeans_model.labels_, cmap='rainbow' )
plt.scatter(reduced_data_pca[:,1],reduced_data_pca[:,2], c=kmeans_model.labels_, cmap='rainbow' )

# This is to match the label name with the outcome labels from kmeans
from scipy.stats import mode
clusters=kmeans_model.labels_
labels = np.zeros_like(clusters)
for i in range(3):
    mask = (clusters == i)
    labels[mask] = mode(labels_true[mask])[0]

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_true, labels)
cm

import seaborn as sns; sns.set()  # for plot styling
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=[0,1,2],
            yticklabels=[0,1,2])
plt.ylabel('true label')

# Supervised-Learning
# Divide your dataset to 80% training and 20% validation
# Build a Logistic regression model for the reduced IRIS dataset
# What is the training accuracy
# What is the validation accuracy
# Form the confusion matrix


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reduced_data_pca, iris_data_clean.iloc[0:,[4]],test_size=0.20)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)

predictions = logisticRegr.predict(X_test)
# Use score method to get accuracy of model
score = logisticRegr.score(X_test, y_test)
# extra question
print(score)


# Form the confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# heatmap
import seaborn as sns; sns.set()  # for plot styling
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=range(3),
            yticklabels=range(3))
plt.ylabel('true label')
plt.xlabel('predicted label');
