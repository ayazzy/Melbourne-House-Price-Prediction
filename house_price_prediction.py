import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

"""
STEP BY STEP HOUSE PRICE PREDICTION MODEL -- Using Decision Tree

1- use pandas to read the csv file that you will be utilizing into a variable

2- drop the non-existent values to be able to work with the data (in the future this will not be the case but since
    I am a beginner this is what I am doing for the time being)
    
3- Select the target value that you are trying to predict aka. the "y" value

4- Select the columns aka. the features that you will be using to make your predictions (the "X" value)

5- Divide your data into training and validation groups using the train_test_split method

6- Build the model -- which in this example our model is a DecisionTreeRegressor

7- Fit the training data into your model 

8- Predict the target value that you wanted to predict using the ".predict(val_X)"

9- Compare the predictions between the real answer and the answer your model came up with using Model Validation formula
    in this example we are using Mean Absolute Error (MAE)
"""

# Doing exploratory data analysis (EDA)
data_path = "/Users/ayazvural/Desktop/Melbourne_House_Data/melb_data.csv"
house_data = pd.read_csv(data_path)
house_data.dropna(axis=0)

# selecting target value
y = house_data.Price

# Selecting features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = house_data[melbourne_features]

# Splitting data into different sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


# Finding the best possible leaf size to find a balance between underfitting and overfitting
possible_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in possible_max_leaf_nodes}


# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(scores, key=scores.get)


# Building the Model
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
final_model.fit(train_X, train_y)


# Model Validation
value_predictions = final_model.predict(val_X)
print("The (in)accuracy of our model is: ", mean_absolute_error(val_y, value_predictions))
