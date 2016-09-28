# we'll do regression with stock prices

import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
# preoprocessing to scale our data (-1 .. 1)
# cross_validation to create our training and testing samples (shuffles data to no have bias)
# support vector machine (to do regression)
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

# high and low nos da la volatidad del día
# open price es como empieza
# close es como cerró y podemos ver si subió o bajo

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# high-low %
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

# % in change troughout the day
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# new dataframe with cols that we really care
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# queremos obtener el precio del futuro (regression) por lo que Adj. Close es nuestro
# label ($)
forecast_col = 'Adj. Close'

# fill with a not a number df.fillna(), pero usamos -99999 porque en ML no usamos NaN
df.fillna(-99999, inplace=True)

# queremos establecer el tamaño del forecast 30 días approx
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

# le aplicamos un shift a la columna a medir (en este caso Adj. Close) para
# variarla por 10 días al futuro
df['label'] = df[forecast_col].shift(-forecast_out)

# elimina los NaN
df.dropna(inplace=True)

# definimos x y y
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# we scale X before feeding to classifier
# scale along the old values
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) # 20% of the data

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train) # train
accuracy = clf.score(X_test, y_test) # to test squared error 

print(accuracy)

# Let's use SVM instead of LinearReg (in this case it's less accurate)
clf = svm.SVR()
clf.fit(X_train, y_train) 
accuracy = clf.score(X_test, y_test) # squared error

print(accuracy)

