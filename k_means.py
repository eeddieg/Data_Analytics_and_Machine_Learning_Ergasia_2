import xlrd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, completeness_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


sns.set()
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
##  K-MEANS  ##
###############
X = StandardScaler().fit_transform(X_)

pca = PCA()
pca.n_components = 2 
X_reduced = pca.fit_transform(X)
print("original shape:   ", X_.shape)
print("transformed shape:", X_reduced.shape)

y = np.ravel(NSR.to_numpy()) # NSP
# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X_reduced, y, test_size=0.3, random_state=50)

clusters = 3
kmeans = KMeans(init="k-means++", n_clusters=clusters, random_state=20,
 n_init=4)
y_pred = kmeans.fit_predict(X_train)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred, cmap="gnuplot")
plt.title("K-Means 3-clustering")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='navy', s=200, alpha=1)
plt.show()

###################################################################

X = StandardScaler().fit_transform(X_)

pca = PCA()
pca.n_components = 2 
X_reduced = pca.fit_transform(X)
print("original shape:   ", X_.shape)
print("transformed shape:", X_reduced.shape)

y = np.ravel(FHR.to_numpy())  # FHR
# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X_reduced, y, test_size=0.3, random_state=50)

clusters = 10
kmeans = KMeans(init="k-means++", n_clusters=clusters, random_state=20,
 n_init=4)
y_pred = kmeans.fit_predict(X_train)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred, cmap="gnuplot")
plt.title("K-Means 10-clustering")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='navy', s=200, alpha=1)
plt.show()
