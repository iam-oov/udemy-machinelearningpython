
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# Dividir el dataset en conjunto de entrenamiento 
# y de testing


url = 'https://raw.githubusercontent.com/joanby/python-ml-course/master/datasets/ads/Advertising.csv'
data = pd.read_csv(url)

a = np.random.randn(len(data))

plt.hist(a)
plt.savefig('f.png')

check = (a<0.8)
training = data[check]
testing = data[~check]

lm = smf.ols(formula='Sales~TV+Radio', data=training).fit()
print (lm.summary())

# Modelo:
# Sales = 2.9089 + 0.0439*TV + 0.2005*Radio

# Validacion del modelo con el conjunto de testing
sales_pred = lm.predict(testing)

SSD = sum((testing['Sales']-sales_pred)**2)
print (SSD)
RSE = np.sqrt(SSD/(len(testing)-2-1))
print (RSE)
sales_mean = np.mean(testing['Sales'])
error = RSE/sales_mean
print (error)