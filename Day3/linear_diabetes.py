from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True, as_frame=True)
diabetes_X = pd.DataFrame(diabetes_X['bmi'])

x_train = diabetes_X[:-20] # Everything but the last 20 values
x_test = diabetes_X[-20:] # The last 20 values
y_train = diabetes_y[:-20]
y_test = diabetes_y[-20:]

# Later we will use this instead
# x_train, x_test = train_test_split(diabetes_X, test_size=0.2)
# y_train, y_test = train_test_split(diabetes_y, test_size=0.2)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test) # Predict the y-values for our test-data


# Plot our regression-line
plt.plot(x_test, y_pred)
#plot this against our actual y-values for the test-data
plt.scatter(x_test,y_test)
plt.show()

print(model.predict([[0.1],[-0.1]]))
