import xlrd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import itertools
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import where, unique

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

pca = PCA()
pca.n_components = 2
X = pca.fit_transform(X)
print("original shape:   ", x_train.shape)
print("transformed shape:", X.shape)

colors = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# define the model

# model = AgglomerativeClustering(n_clusters=3)
model = AgglomerativeClustering(n_clusters=10)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], color=next(colors))
# show the plot
pyplot.show()
