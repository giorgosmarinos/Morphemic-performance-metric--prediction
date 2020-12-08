import os, time
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle


class Trainer():
    def __init__(self, url_file, target, application, features_list):
        self.url_file = url_file
        self.target = target
        self.features_list = features_list
        self.application = application
        self.ml_file_link = None

    def load_data(self):
        # Check if we received a file that does not exist
        if os.path.isfile(self.url_file):
            pass
        else:
            raise ValueError("Sorry, find does not exists.")
        name, extension = os.path.splitext(self.url_file)
        # check if it is not a csv file
        if extension != '.csv':
            raise ValueError("Oops!  That was no a csv file.  Try again...")
        else:
            pass
        # check if it is an empty file
        if os.stat(self.url_file).st_size == 0:
            raise ValueError("Sorry!  That file is empty.  Try again...")
        else:
            pass
        # Load the file
        data = pd.read_csv(self.url_file, names=self.features_list)  # ή columns ή names
        # df_3 = pd.DataFrame(X,columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        # 'TAX', 'PTRATIO', 'B', 'LSTAT'])
        # df_3['MEDV'] = y
        return data

    def choose_variable_based_on_importance(self):
        pass

    def missing_values_imputer(self, data):
        # data = self.load_data()

        # TODO ---> make a configuration column
        # TODO ---> use the missing values imputation based on configuration

        if data.isna().any().any() == False:
            pass
        elif data.isna().any().any() == True:
            names = data.columns
            # example
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(data.values)
            data = imp.transform(data.values)
            data = pd.DataFrame(data, columns=names)
        return data

    def normalization(self, data):
        # data = self.load_data()
        scaler = StandardScaler()
        names = data.columns
        scaler.fit(data)
        data = scaler.transform(data)
        data = pd.DataFrame(data, columns=names)
        return data

    def encode_categorical_values(self, data):
        # data = self.load_data()
        a = []
        j = 0
        for i in data.dtypes:
            if i == np.dtype('str') or i == np.dtype('object'):
                a.append(data.dtypes.index[j])
                j += 1
            elif i != np.dtype('str') or i != np.dtype('object'):
                j += 1

        # TODO ----> I shall add more encoder functions, not only Ordinal Encoder

        oe = OrdinalEncoder()
        categorical_data = data[a]
        names = categorical_data.columns
        oe.fit(categorical_data)
        categorical_data = oe.transform(categorical_data)
        categorical_data = pd.DataFrame(categorical_data, columns=names)
        # replace categorical with encoded data
        for i in a:
            data[i] = categorical_data[i].replace(',', '-')

        return data

    def train_and_test_split(self, data):
        # data = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=[self.target]),
                                                            data[self.target], test_size=0.30)
        return X_train, X_test, y_train, y_test

    def train(self):
        data = self.load_data()
        data = self.missing_values_imputer(data)
        data = self.encode_categorical_values(data)
        data = self.normalization(data)

        # TODO ----> I shall put also here all possible regressor that I want to have

        # RandomForestRegressor
        self.regressor = RandomForestRegressor(random_state=0)
        X_train, X_test, y_train, y_test = self.train_and_test_split(data)
        self.regressor.fit(X_train, y_train)
        # cross_val_score(regressor, X_train, y_train)
        pred = self.regressor.predict(X_test)
        pred_train = self.regressor.predict(X_train)
        # R2_test
        print('R2_test: ', r2_score(y_test, pred))
        # R2_train
        print('R2_train: ', r2_score(y_train, pred_train))
        # MSE
        print('Mean Squared Error', mean_squared_error(y_test, pred, squared=True))
        # RMSE
        print('Mean Error', mean_squared_error(y_test, pred, squared=False))
        # MAE
        print('Mean Absolute Error', mean_absolute_error(y_test, pred))

    def getMLFile(self):
        self.train()
        # To filename να ειναι το ονομα της εφαρμογης
        filename = self.application + '.sav'
        pickle.dump(self.regressor, open(filename, 'wb'))
        self.ml_file_link = pickle.load(open(filename, 'rb'))
        path = os.path.dirname(os.path.realpath('self.ml_file_link'))
        # Now we can choose either to return the file or the path where the file it is stored
        return {'file': self.ml_file_link, 'application': self.application, 'filename': filename}


class Prediction():
    def __init__(self, application, features_dict, target):
        self.application = application
        self.features_dict = features_dict
        self.target = target
        self.prediction = None

    def predict(self):
        # Dont use target variable in feature_list
        # check the name of application
        # load the pickle file based on application name
        # Print (return) error if the file dont exist. This means that we dont have trained this model yet
        # 1 -> λαθος ονομα
        # 2 -> σωστο ονομα αλλα η εφαρμογή δεν εχει γινει ακομα evaluated (the application
        # was not evaluated yet or the invalid name)
        # Σε περιπτωση που υπάρχει θα λαμβάνουμε το feature_list και το target για να γίνει
        # το prediction
        new_sample = pd.DataFrame.from_dict(self.features_dict, orient='columns').values
        print(new_sample)
        self.name = self.application + '.sav'
        self.trained_algorithm = pickle.load(open(self.name, 'rb'))
        self.prediction = self.trained_algorithm.predict(new_sample)
        return self.prediction

    def getPrediction(self):
        return self.prediction
