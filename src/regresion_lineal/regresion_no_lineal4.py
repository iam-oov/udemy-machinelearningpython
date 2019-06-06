# transformacion de variables para conseguir una 
# relacion no lineal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def regresion_validation(x_data, y, y_pred):
	SSD = np.sum((y - y_pred) ** 2)
	RSE = np.sqrt(SSD/(len(x_data)-1))
	y_mean = np.mean(y)
	error = RSE/y_mean
	return SSD, RSE, error

def create_chart_prediction(x, y, x_pred, name):
	plt.plot(x, y, 'ro')
	plt.plot(x, x_pred, color="blue")
	plt.savefig(name)



data = pd.read_csv('../python-ml-course/datasets/auto/auto-mpg.csv')

# eliminar los NA
data['mpg'] = data['mpg'].dropna()
data['horsepower'] = data['horsepower'].dropna()

plt.plot(data['horsepower'], data['mpg'], 'ro')
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.title('HP vs MPG')
plt.savefig('exponencial.png') # exponencial o cuadratica en forma de parabola 

# primer test: modelo de refresion lineal
# mpg = a + b * horsepower

# Modelo de regresion cuadratico
# mpg = a + b * horsepower ** 2

x = data['horsepower'].fillna(data['horsepower'].mean())
x_data = x[:, np.newaxis]

x2_data = x**2
x2_data = x2_data[:, np.newaxis]

y = data['mpg'].fillna(data['mpg'].mean())


lm = LinearRegression().fit(x_data, y)
lm2 = LinearRegression().fit(x2_data, y)

#create_chart_prediction(x, y, lm.predict(x_data), name="regresion.png")


print ('(R2) Score LM:', lm.score(x_data, y))
print ('(R2) Score LM cuadratico:', lm2.score(x2_data, y))

print(regresion_validation(x_data, y, lm.predict(x_data)))
print(regresion_validation(x2_data, y, lm2.predict(x2_data)))


# hasta este punto los dos modelos anteriores no nos dan
# el mejor de los resultados asi que optaremos por combinarlos


# Modelo de regresion cuadratico + lineal
# mpg = a + b * horsepower + c * horsepower**2

poly = PolynomialFeatures(degree=2)
x_data = poly.fit_transform(x[:, np.newaxis])
lm = LinearRegression()
lm.fit(x_data, y)
print ('(R2) Score poly:', lm.score(x_data, y))


# el score es muy bueno que podemos usar este modelo
# mpg = intercept * coef[1]*hp +  coef[2]*hp**2 (por ser grado 2)

intercept = lm.intercept_
coef = lm.coef_
print (regresion_validation(x_data, y, lm.predict(x_data)))




# El problema de los outliers
print ('------------')
print ('problema de outliers')

x = data['displacement'].fillna(data['displacement'].mean())
x_data = x[:, np.newaxis]

lm = LinearRegression().fit(x_data, y)

print ('(R2) Score', lm.score(x_data, y))
create_chart_prediction(x, y, lm.predict(x_data), name='outliers.png')


# eliminar los indeces que salen de la regla
print (data[(data['displacement']>300)&(data['mpg']>20)])

data_clean = data.drop([395, 258, 305, 372])
x = data_clean['displacement'].fillna(data_clean['displacement'].mean())
x_data = x[:, np.newaxis]
y = data_clean['mpg'].fillna(data_clean['mpg'].mean())

lm = LinearRegression().fit(x_data, y)
print (lm.score(x_data,y))