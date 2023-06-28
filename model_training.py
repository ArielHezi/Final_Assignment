import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
import pickle
from madlan_data_prep import prepare_data
from sklearn.model_selection import train_test_split


data = pd.read_excel("output_all_students_Train_v10.xlsx")
dataset = data.copy()
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=0)
train_data = prepare_data(train_data)
test_data = prepare_data(test_data)

# Separate the features and target for training data
X_train = train_data.drop(['price'], axis=1)
y_train = train_data['price']

# Separate the features and target for test data
X_test = test_data.drop(['price'], axis=1)
y_test = test_data['price']

categorical_columns = ['type', 'City', 'entranceDate', 'city_area', 'Street']
numerical_columns = ['room_number', 'Area', 'floor', 'total_floors', 'description']

# Create the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Create the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('E', ElasticNet())
])

# Define the parameter grid for hyperparameter search
param_grid = {
    'E__alpha': [0.05, 0.06, 0.07, 0.09, 0.1, 0.5, 1.0, 5.0, 5.5],
    'E__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Create the GridSearchCV object with 10-fold cross-validation
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10)

# Fit the GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
pickle.dump(best_model, open("trained_model.pkl","wb"))


