# Machine_learning_Capstone_iii
Applying clustering techniques and PCA on the UsArrest data set
## Table of Content
### 1. Introduction
### 2. Technologies
### 3. Layout of the of the Jupeyter Notebook
### 4. Conclusion

### Introduction
The objective of the project was to apply an three technologies of machine learning, namely clustering techniques after I have to  applyied the Pricipal Component analysis. I wanted to see how would this affect the performance of the clustering model.

### Technologies
The Technologies employed in this project are at follows:
### Libraries used
import numpy as np
import pandas as pd
#### Visualisazation libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#### sklearn libraries that i used in the project
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

### Layout of the Jupyter NoteBook
I first start with an Exploratory data Analysis, check for missing data and preprocess the data set, luckly the data set was clean and contain no missing values. Plot a corrolation heatmap to visialize the relationships of the instances in the data set.
The after I initiated the machine learning tools, namely Kmeans and the agglomerative clustering, fit the data.
After feeting the data i evaluated the performance of the models using the Silhouette score. the results that i got suggested that these model were not a good fit for the dataset.
One reason that i think led to such results, due to the fact that the data set contains continuous data and i thin a regression model would be best fitted for the data set.
### Conclusion 
The clustering models were not a good fit for thr data set. i think a model that would best fit the data set is a regression model as the data set containes continuous values.
