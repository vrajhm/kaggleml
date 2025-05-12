from sklearn.tree import DecisionTreeRegressor
import pandas as pd

melb_housing_path = "MelbHouse/melb_data.csv"

melb_data = pd.read_csv(melb_housing_path)

y = melb_data.Price
X = melb_data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]

melb_model = DecisionTreeRegressor(random_state=1)
melb_model.fit(X, y)

print(X.head())
print(pd.DataFrame(melb_model.predict(X.head())))