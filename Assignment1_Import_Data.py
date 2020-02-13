# %%

import pandas as pd

#locating the dataset, getting rid of any NA values, and printing the first 10 rows
dataset = pd.read_csv('/content/drive/My Drive/housing.csv')
dataset = dataset.dropna()
print("Here are the first ten rows of the dataset:")
dataset.head(10)


#Plotting the dataset%%
dataset.plot(subplots=True)