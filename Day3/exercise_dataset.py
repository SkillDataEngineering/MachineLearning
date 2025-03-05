from sklearn import datasets
import pandas as pd
from sklearn import train_test_split
from sklearn import Linear_regression

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y= True, as_frame= True)

print(diabetes_X.describe())

print(diabetes_X.corr(numeric_only=True))

diabetes_bmi_X = pd.DataFrame(diabetes_X['bmi'])

X_train, X_test, y_train, y_test = train_test_split(diabetes_bmi_X, diabetes_y,test_size=0.05, random_state=42)

model = Linear_regression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)