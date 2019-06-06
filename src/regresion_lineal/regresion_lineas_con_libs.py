from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd
from sklearn.linear_model import LinearRegression


url = 'https://raw.githubusercontent.com/joanby/python-ml-course/master/datasets/ads/Advertising.csv'
data = pd.read_csv(url)

feature_cols = ['TV', 'Newspaper', 'Radio']
x = data[feature_cols]
y = data['Sales']

estimator = SVR(kernel='linear')
selector = RFE(estimator, 2, step=1)
selector = selector.fit(x, y)

print (selector.support_)
print (selector.ranking_)

x_pred = x[['TV', 'Radio']]

lm = LinearRegression()
lm.fit(x_pred, y)

print (lm.intercept_)
print (lm.coef_)
print (lm.score(x_pred, y))