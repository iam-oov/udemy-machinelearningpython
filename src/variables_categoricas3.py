import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

url = '../python-ml-course/datasets/ecom-expense/Ecom_Expense.csv'
df = pd.read_csv(url)

print (df.head())

dummy_gender = pd.get_dummies(df['Gender'], prefix='Gender')
dummy_city_tier = pd.get_dummies(df['City Tier'], prefix='City')

# merge the new data with dataframe
columns_names = df.columns.values.tolist()

df_new = df[columns_names].join(dummy_gender).join(dummy_city_tier)
columns_names = df_new.columns.values.tolist()


feature_cols = ['Monthly Income', 'Transaction Time', 
				'Gender_Female', 
				'City_Tier 1', 
				'Record'
				]

x = df_new[feature_cols]
y = df_new['Total Spend']

lm = LinearRegression()
lm.fit(x, y)

intercept = lm.intercept_
coef = lm.coef_

print(lm.score(x,y))
# El modelo puede ser escrito como:

pre_resultado = 0
for i,v in enumerate(feature_cols):
	pre_resultado +=  df_new[v] * coef[i]

df_new['prediction'] = lm.intercept_ + pre_resultado
df_new['prediction2'] = lm.predict(pd.DataFrame(df_new[feature_cols]))


SSD = np.sum((df_new['prediction2'] - df_new['Total Spend'])**2)
RSE = np.sqrt(SSD/(len(df_new)-len(feature_cols)-1))
sales_mean = np.mean(df_new['Total Spend'])
error = RSE/sales_mean

print ("Error: {}".format(error*100))
print ("RSE: +/- {}".format(RSE))

# eliminar variables dummy redundantes