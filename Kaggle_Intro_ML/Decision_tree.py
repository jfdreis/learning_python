#In this file I want to use Pandas and I am doing Kaggle's course: Intro to machine learning

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



#house prices for Iowa

#iowa_file_path = 'C:\Users\josev\Documents\Github\learning_python\Linear_logistic_regression\train.csv'

iowa_data= pd.read_csv('train.csv')
iowa_data= iowa_data.drop("Id", axis=1)

#################

#trying things

###################
"""
print(iowa_data.head())
print(iowa_data.describe())
print(iowa_data.shape)
print(iowa_data.dtypes)
print(iowa_data.isnull().sum())
print(iowa_data.duplicated().sum())

"""
#print(iowa_data.columns) #get the header

############
#compute some data

##########
#get the  average size
# Just need to consider the corresponding collumn

# Average are of the houses
"""
lot_area=iowa_data.loc[:,"LotArea"]
print(lot_area)
avg=np.average(lot_area)
print(avg)
avg_lot_size=round(home_data["LotArea"].mean(),0) #Better this way, no need for np

#year of oldest and newest houses"

newest_home_age = 2023- iowa_data["YearBuilt"].max()

year_built=iowa_data.loc[:,"YearBuilt"]
print(min(year_built))
print(max(year_built))
"""

#############################

# Now I want to develop a model for this data

#############################

y=iowa_data.SalePrice
#print(y)

#Now I want to use prediction features

prediction_features=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X=iowa_data[prediction_features]
#print(X.describe())
#print(X.head())


########################

#Using the decision tree algorithm

############

iowa_model=DecisionTreeRegressor(random_state=1) 

#Fit the Model
iowa_model.fit(X,y)

#print the prediction

prediction=iowa_model.predict(X)
#print(prediction)



#Print only the first 5 cases

"""
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))
"""

#Calculating the mean absolute error


error=mean_absolute_error(y,prediction)
print(error)

#######################

#Spliting the data changes the error


##############

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
iowa_model=DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X,train_y)
prediction=iowa_model.predict(val_X)
error=mean_absolute_error(val_y,prediction)
print(error)

#####################

#Here we will see how the mean absolute error changes when we consider more leaf leavel for the decision tree algotithm

####################
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

#print(scores)
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(scores, key=scores.get) # I have to understand this better, I get what it does. Not how it is working.
print(best_tree_size)


# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)

# fit the final model and uncomment the next two lines
final_model.fit(X,y)


######################

#I could compare this with the previous model


###############################

######################

#Using random forest

#############

rf_model = RandomForestRegressor(random_state=1)

# fit model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of Random Forest model on the validation data
rf_model_preds = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y,rf_model_preds)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))