from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

classifier = LinearRegression()
classifier.fit(X, y)
guess = classifier.predict(X)
error = mean_squared_error(y, guess)
