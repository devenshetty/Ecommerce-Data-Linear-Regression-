import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading the dataset
dataset = pd.read_csv("Ecommerce Customers")

#exploratory data analysis
sns.set_palette("PRGn")
sns.set_style("darkgrid")
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=dataset)

sns.set_palette("Accent")
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=dataset)

#comparing time on app and length of membership
sns.set_palette("GnBu_d")
sns.jointplot(x="Time on App", y="Length of Membership", data=dataset, kind="hex")

#identifying the best features to consider for the analysis by observing the correlation
sns.set_palette("Set2")
sns.pairplot(dataset)
#based on the plots, length of membership seems to be highly correlated with yearly amount spent

#linear plot for the above two attributes
sns.set_palette("PiYG")
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=dataset)

#splitting the data for training and testing
X = dataset[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = dataset["Yearly Amount Spent"]

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

#training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train) 

print("Coefficients:", lm.coef_)

#testing the model
y_pred = lm.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values of 'Yearly Amount Spent'")
plt.ylabel("Predicted Values 'Yearly Amount Spent'")

#evaluating the performance of the model
from sklearn import metrics
print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#plotting a histogram of residuals
sns.displot(y_test-y_pred, bins=50, kde = True)
#the histogram looks like a normal distribution and hence it can be concluded that the model performed well


df = pd.DataFrame(lm.coef_, X.columns, columns = ["Coefficient"])
print(df)

#result interpretation
'''
Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.
'''






