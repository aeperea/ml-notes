# we'll do regression with stock prices
import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
# preoprocessing to scale our data (-1 .. 1)
# cross_validation to create our training and testing samples (shuffles data to no have bias)
# support vector machine (to do regression)
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle # like a file that saves the classifier

style.use('ggplot')

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

df['label'] = df[forecast_col].shift(-forecast_out) # shift, días al futuro
X = np.array(df.drop(['label'], 1))                 # definimos x y y
X = preprocessing.scale(X)                          # we scale X before feeding to classifier scale along the old values
X = X[:-forecast_out]
X_lately = X[-forecast_out:]                        # we don't have a Y value on these:


df.dropna(inplace=True);
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) # 20% of the data

# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train) # train

# # we want to save the classifier to not retrain it, so we need to use pickle
# # we don't want to retrain it everytime
# with open('linearregression.pickle','wb') as f:
#     pickle.dump(clf, f) # dumps the classifier

#to use the saved classifer (read) 
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test) # to test squared error 
# print(accuracy)


# we need to predict based on the X data (single_value or array of values to predict)
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

# agregamos la columna de forecast y le damos la fecha de los siguiente días para podedr hacer el plot bien
df['Forecast'] = np.nan # entire column as NaN
last_date      = df.iloc[-1].name # last record
last_unix      = last_date.timestamp()
one_day        = 86400
next_unix      = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i] # .loc references the index


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show();
