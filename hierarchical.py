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

Y = FHR.to_numpy()
# Y = NSR.to_numpy()


# # feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X_)

pca = PCA()
pca.n_components = 2
X = pca.fit_transform(x_scaled)
print("original shape:   ", X_.shape)
print("transformed shape:", X.shape)

neign_num = 5
algorithm  = "ball_tree"
# algorithm  = "auto"
neigh = NearestNeighbors(n_neighbors=neign_num, algorithm=algorithm).fit(X)
distances, indices = neigh.kneighbors(X, n_neighbors=neign_num)
print(distances)

graph = neigh.kneighbors_graph(X).toarray()
print(graph)

