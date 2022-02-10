from matplotlib.colors import ListedColormap
import xlrd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import MeanShift, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, completeness_score

plt.style.use('ggplot')
warnings.filterwarnings('ignore')


################
# Load dataset #
################

FILE_NAME = "./Dataset/CTG.xls"
# method for reading .xls. Created due to floating error of read_excel()


def readExcelFile(filename):
    workbook = xlrd.open_workbook(filename)
    sheet_names = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheet_names[2])

    content = list()
    for i in range(worksheet.nrows):
        content.append(worksheet.row_values(i))

    headers = content[0]
    df = pd.DataFrame(content[1:], columns=headers)

    return df


df1 = readExcelFile(FILE_NAME)


#########################
# Cleaning the  dataset #
#########################

cols = ["LB", "AC", "FM", "UC", "ASTV", "MSTV", "ALTV", "MLTV", "DL", "DS", "DP",  "Width", "Min",
        "Max", "Nmax", "Nzeros", "Mode",	"Mean", "Median", "Variance", "Tendency", "CLASS", "NSP"]
df = df1.filter(items=cols, axis="columns")[1:-3]
# print(df.isnull().sum().sum())
df = df.dropna()
# data type to float16, saving memory
df = df.astype("float16")

# separate 2 last columns as validation set
cols = cols[:-2]
X_ = df.filter(items=cols, axis="columns")  # Main dataset
FHR = df["CLASS"].astype("int")  # FHR
NSR = df["NSP"].astype("int")   # NSR


###############
## MeanShift ##
###############

X = StandardScaler().fit_transform(X_)
# X = MinMaxScaler().fit_transform(X_)

pca = PCA()
pca.n_components = 2
X_reduced = pca.fit_transform(X)
print("original shape:   ", X_.shape)
print("transformed shape:", X_reduced.shape)


ms = MeanShift()
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters = len(np.unique(labels))
print("Number of estimated clusters: ", n_clusters)

# colors = 5*["r.", "g.", "b." ,"c." ,"o.", "k.", "y.", "m."]
colors = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

 
for i in range(len(X_reduced)):
  plt.plot(X[i][0], X[i][1], color=next(colors), markersize = 10)
#   plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker = "x", s = 200, linewidths = 5, zorder = 10)


def displayResults(test, pred):
  target_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
  print(classification_report(test, pred, target_names=target_names))
  accuracy = accuracy_score(test, pred)
  precision = precision_score(test, pred, average='weighted')
  f1Score = f1_score(test, pred, average='weighted')
  print("Accuracy  : {}".format(accuracy))
  print("Precision : {}".format(precision))
  print("f1Score : {}".format(f1Score))
  c_matrix = confusion_matrix(test, pred)
  print("\nConfusion matrix")
  print(c_matrix)

y_true = FHR.to_numpy()
unique = np.unique(y_true)
length = len(unique)

displayResults(labels, y_true)


