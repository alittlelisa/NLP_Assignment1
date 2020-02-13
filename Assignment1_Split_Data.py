from Assignment1_Import_Data import dataset
from sklearn.model_selection import train_test_split

#getting the X and Y values
X = dataset.loc[:, 'longitude':'median_income']
Y = dataset['median_house_value']

#splitting the data 70:30 for training and testing respectively
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2003)
print("There are " + str(x_train.size) + " training entries and "+
      str(x_test.size) + " testing entries!")

#converting the testing and training sets to numpy arrays
x_train_np = x_train.to_numpy()
y_train_np = y_train.to_numpy()

x_test_np = x_test.to_numpy()
y_test_np = y_test.to_numpy()