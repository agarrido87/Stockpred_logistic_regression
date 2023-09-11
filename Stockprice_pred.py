import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
import pandas_ta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load local csv file as a dataframe
df_tsla = pd.read_csv('C:/Users/agarr/Desktop/Atom_Scripts/Stock Regression/TSLA.csv')

# Check that the data has been loaded as a dataframe
print(df_tsla)

# Get stats summary
print(df_tsla.describe())

# Reindex data just chnged this from True to false||||Make sure to change back the inplace to true if it stops working
df_tsla.set_index(pd.DatetimeIndex(df_tsla['Date']), inplace=True)
print(df_tsla)

# Eliminate other columns except for adj close
df_tsla = df_tsla[['Adj Close','Date']]

# Print info
print(df_tsla.info())
print(df_tsla)

# Create line graph based on date and adj close
df_tsla.plot(x='Date', y='Adj Close', kind='line')
plt.show()

# Adding Exponential moving average
df_tsla['EMA_10'] = df_tsla['Adj Close'].rolling(window=10).mean()

# View data results to check for NAs
print(df_tsla)

df_tsla = df_tsla.iloc[10:]

# View new dataset
print(df_tsla.head(10))

# Plot with Exponential moving average displaying adj close and EMA_10
df_tsla.plot(x='Date', y=['Adj Close', 'EMA_10'])
plt.legend(loc='lower right')
plt.xlabel('Date')
plt.ylabel('Adj Close')
plt.show()

#added Date here to see what it does... but should delete
# Split data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(df_tsla[['Adj Close']], df_tsla['EMA_10'], test_size=.2)

# Create Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Use model to make predictions
y_pred = model.predict(X_test)

# Print relevant metrics
print("Model Coefficients:", model.coef_)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))

#For changing line 20 into false now we lost the date on test data. We should find a way to add date into test data
# Get the date column from the test data
date_column = X_test.index


# Plot predicted vs real value
plt.figure(figsize=(4, 3))
plt.scatter(date_column, y_pred)
plt.show()

# Print predicted and actual values
for i in range(len(X_test)):
    print(date_column[i], y_pred[i])

###NEED TO FIX THE EXTRA SPACING ON DATA FRAME THIS IS WHY IT DOESN'T wirte on the CSV filePath
###This was fixed by adding index false at the end and specifying where to export csv file
results_df = pd.DataFrame({'Date': date_column, 'Actual': y_test, 'Predicted': y_pred})


print(results_df)
#lets do a quick check of column types
#results_df.dtypes

#new code to try to export to csv This works and fixes index error by adding the index False
results_df.to_csv(r'C:\Users\agarr\Desktop\Atom_Scripts\Stock Regression\tesla_prediction.csv', index=False)



####The code below is different ways to export to csv but I was getting errors because of indexing
###test below to save cvs in path I want
# name of csv file
#fields = ['Date', 'Actual', 'Predicted']
#filename = "tesla_prediction.csv"

#filePath = "C:/Users/agarr/Desktop/Atom_Scripts/Stock Regression"

# writing to csv file
#path = os.path.join(filePath, filename)
#with open(path, 'w', newline='') as csvfile:
    # creating a csv dict writer object
    #writer = csv.DictWriter(csvfile, fieldnames=fields)

    # writing headers (field names)
    #writer.writeheader()

    # writing data rows
    #writer.writerows(results_df)
