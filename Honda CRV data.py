""" IMPORT THE LIBRARIES """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

""" LOAD THE DATA """

raw_data = pd.read_csv("C:/Users/sivam/OneDrive/Projects/Multi Linear Regression/honda/Honda CRV Data_042015.csv")
raw_data.describe(include='all')

raw_data.isnull().sum()
data_nonull = raw_data.dropna(axis = 0)
data_nonull.describe(include = 'all')

""" DISTRIBUTION"""

sns.distplot(data_nonull['Mileage'])
q = data_nonull['Mileage'].quantile(0.99)
data_1 = data_nonull[data_nonull['Mileage']<q]

sns.distplot(data_1['Model_Year - 1999'])
q = data_1['Model_Year - 1999'].quantile(0.01)
data_2 = data_1[data_1['Model_Year - 1999']>q]

data_cleaned = data_2.reset_index(drop = True)

""" LINEARITY """
f,(ax1, ax2) = plt.subplots(1,2, sharey = True, figsize = (15,3))
ax1.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax1.set_title('Price vs Mileage')
ax2.scatter(data_cleaned['Model_Year - 1999'], data_cleaned['Price'])
ax2.set_title('price vs Model Year')
plt.show()

""" MULTICOLLINEARITY """

variables = data_cleaned[['Mileage', 'Model_Year - 1999']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
vif

""" LINEAR REGRESSION MODEL """
targets = data_cleaned['Price']
inputs = data_cleaned.drop(['Price'], axis = 1)
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size =0.2, random_state =12)
reg = LinearRegression()
reg.fit(x_train, y_train)

""" CHECK """
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)
reg.score(x_train,y_train)

""" weights """

reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values, columns = ['features'])
reg_summary['weights'] = reg.coef_
reg_summary
