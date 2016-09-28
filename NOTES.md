# Machine Learing Youtube course

https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

## Simple Linear Regression
### Video 2 - Regression Intro (https://youtu.be/JcI5Vnw0b2c) 

Features - attributes

Podemos usar Quandl para sacar datasets desde quandl.com. Sólo buscamos la
inforamción que necesitamos y usamos el Quandl Code para extraer los datos.

`df = quandl.get('WIKI/GOOGL')`

Queremos features que no se repitan porque corremos riesgo de overfitting

### Video 3 - Regression Features and Labels (https://youtu.be/lN5jesocJjk) 
Labels - lo que queremos determinar (en este caso, el precio)

Para analizar el dataframe (pandas):

`df.head()` (primeros n elementos)
`df.tail()` (últimos n elementos)

### Video 4 - Regression Training and Testing (https://youtu.be/r4mwkS2T9aI)

Usamos numpy para manejar todo como arrays ->
`y = np.array(df['label'])`

Escalamos X antes de darselo al clasiffier
`X = preprocessing.scale(X)`

Es importante notar que si escalamos nuestros datos debemos escalar cualquier
nuevo dato contra el cual querramos comparar, así como reescalar los datos que
ya tenemos. (a veces no vale la pena escalar).

Va a tomar todas nuestras feats y labels, las mezcla y nos devuelve trains para
X y Y, y tests también
```
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) # 20% of the data
```

Ahora vamos a usar un classifier, en este caso LinearRegression
`clf = LinearRegression()`

Le hacemos fit para los trains
`clf.fit(X_train, y_train)`

Y usamos los datos de test para ver su accuracy:

`accuracy = clf.score(X_test, y_test)`

Necesitamos tener un set de test y otro de trains para comparar la habilidad de
neustro algoritmo para a ver si funciona.


Podemos usar threaded algos en algunos con sklearn
(http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

En nuestro caso, al usar LinearRegression le podemos agregar la opción de n_jobs
para repartir la tarea (si el input es -1, usa el máximo posible por el CPU)
`clf = LinearRegression(n_jobs=-1)`


### Video 5 - Regression forecasting and predicting (https://youtu.be/QLVMqwpOLPk)
