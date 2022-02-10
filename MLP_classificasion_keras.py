import h5py
import numpy as np
import pandas as pd
import os
import warnings
import xlrd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings('ignore')

# Read data
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

df1  = readExcelFile(FILE_NAME)
cols = ["LB", "AC", "FM", "UC", "ASTV", "MSTV", "ALTV", "MLTV", "DL", "DS", "DP",  "Width", "Min",
        "Max", "Nmax", "Nzeros", "Mode",	"Mean", "Median", "Variance", "Tendency", "CLASS", "NSP"]
df   = df1.filter(items=cols, axis="columns")[1:-3]
# print(df.isnull().sum().sum())
df   = df.dropna()
# data type to float16, saving memory
df   = df.astype("float16")

# separate 2 last columns as validation set
cols = cols[:-2]
X_   = df.filter(items=cols, axis="columns")  # Main dataset
FHR  = df["CLASS"].astype("int")  # FHR
NSR  = df["NSP"].astype("int")   # NSR

y = FHR.to_numpy()

# #=====Split Dataset=======

xTrain, xTest, yTrain, yTest = train_test_split(X_, y, test_size=0.20, random_state=10)

#=====Feature Scaling======
# columnsToScale = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest  = scaler.transform(xTest)


def getModel():
  model = Sequential()

  # model.add(Dense(100, input_dim=shape, activation=activationFunction))
  # model.add(Dense(75, activation=activationFunction))
  # model.add(Dense(50, activation=activationFunction))
  # model.add(Dense(25, activation=activationFunction))
  # model.add(Dense(11, activation=activation_out))

  model.add(Dense(1000, input_dim=shape, activation=activationFunction))
  model.add(Dense(3000, activation=activationFunction))
  model.add(Dense(25, activation=activationFunction))
  model.add(Dense(11, activation=activation_out))

  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  return model

def predictionToCategorical(pred):
  tmp = np.zeros_like(pred)

  for i in range(len(pred)):
    index_max_val = np.argmax(pred[i])
    tmp[i][index_max_val] = 1

  return tmp

def displayResults(test, pred):
  target_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
  print(classification_report(test, pred, target_names=target_names))
  accuracy = accuracy_score(test, pred)
  precision = precision_score(test, pred, average='weighted')
  f1Score = f1_score(test, pred, average='weighted')
  print("Accuracy  : {}".format(accuracy))
  print("Precision : {}".format(precision))
  print("f1Score : {}".format(f1Score))
  c_matrix = confusion_matrix(test.argmax(axis=1), pred.argmax(axis=1))
  print("\nConfusion matrix")
  print(c_matrix)


# Cross Validation 
folds               = 5       # k as in k-fold
shuffle             = True    # shuffle data
state               = 10      # random_state 
verbose             = 0       # 0/1 -> show progress on console 
epochs              = 100
batch_size          = 4

shape               = xTrain.shape[1]
activationFunction  = 'relu'
activation_out      = "softmax"
loss                = "categorical_crossentropy"
optimizer           = "adam"
metrics             = ['accuracy']

model = getModel()

# skf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=state)
# foldNum = 0
# for train_index, test_index in skf.split(xTrain, yTrain):
#   foldNum += 1
#   print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#   print("Results for fold", foldNum)
#   X_train, X_test = xTrain[train_index], xTrain[test_index]
#   Y_train, Y_test = yTrain[train_index], yTrain[test_index]

#   # one hot encode
#   Y_train = to_categorical(Y_train).astype(int)
#   Y_test = to_categorical(Y_test).astype(int)

#   # # save best model to disk 
#   save_model_name = os.path.join('./models/MLP_classifier.h5')
#   saveBest = ModelCheckpoint(save_model_name, monitor='val_loss', save_best_only=True)
#   earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=2, mode='auto')

#   # train the model
#   # history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.05, callbacks=[saveBest, earlyStopping])
#   history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs,
#     batch_size=batch_size, verbose=verbose, validation_split=0.05, callbacks=[saveBest, earlyStopping])

#   # predict values
#   y_pred = model.predict(X_test)

#   #Converting prediction output to categorical
#   y_pred_cat = predictionToCategorical(y_pred).astype(int)
  
#   # Display results
#   displayResults(Y_test, y_pred_cat)


# model.summary()

####################################################

#####################
#   Evaluate model  #
#####################

# load model
model = load_model('./models/MLP_classifier.h5')
# summarize model.
model.summary()

# evaluate the model
score = model.evaluate(xTest, to_categorical(yTest).astype(int), verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
