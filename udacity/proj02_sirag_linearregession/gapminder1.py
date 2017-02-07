# TODO: Add import statements
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
x = bmi_life_data['BMI']
y = bmi_life_data['Life expectancy']


# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = None
bmi_life_model = LinearRegression()
x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)
bmi_life_model.fit(x, y)
# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = None
laos_life_exp = bmi_life_model.predict([[21.07931],])
print(laos_life_exp)
