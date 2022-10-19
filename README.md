# euprope-railways
railcompanies
import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\Users\Lenovo\Downloads\all_ways.csv",encoding='latin1')

import numpy as np ## numerical(mathematical) calculations

## Understanding the data
data.info()  ## information about the null,data type, memory
data.describe() ## statistical information
data.shape 
data.columns

'''
['type', 'source_station', 'destination_station', 'goods_weight',
       'containers', 'labour_cost', 'transportation_cost']
 '''


## Data cleaning

data.duplicated().sum() ## no duplicates
data.isna().sum() # no null values


# Create dummy variables
train_new = pd.get_dummies(data)
train_new_1 = pd.get_dummies(data, drop_first = True)
# we have created dummies for all categorical columns
###########





########################################
###standardisation
from sklearn.preprocessing import StandardScaler
# Initialise the Scaler
scaler = StandardScaler()
# To scale data
train_std = scaler.fit_transform(train_new_1)
# Convert the array back to a dataframe
train_dataset = pd.DataFrame(train_std)
res = train_dataset.describe()
################################################




##################################
#normalisation

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

train_norm = norm_func(train_dataset)
b = train_norm.describe()

###############################################
train_norm.describe()
###########################################
y=train_norm[3]
X1=train_norm.iloc[:,0:3]
X2=train_norm.iloc[:,4:]
X = pd.concat([X1,X2], axis=1)


###############################

#TRAIN AND TEST


#let's split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)



#now lets build a model using fitting Multilinear Regression to the training set
from sklearn.linear_model import LinearRegression
start_f = LinearRegression() 
start_f.fit(X_train, y_train)



#prediction for the test results
y_pred = start_f.predict(X_test)
y_pred

#prediction for the train results
y_pred1 = start_f.predict(X_train)
y_pred1

#for comarision to check whether the y_test is good or y_pred is good
#we will use r^2 score from the sklearn package
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred) 
score


from sklearn.model_selection import cross_val_score

# Predicting Cross Validation Score the Test set results
cv_linear = cross_val_score(estimator = start_f, X = X_train, y = y_train, cv = 10)
cv_linear

# Predicting R2 Score the Train set results
y_pred_linear_train = start_f.predict(X_train)
r2_score_linear_train = r2_score(y_train, y_pred_linear_train)
r2_score_linear_train

# Predicting R2 Score the Test set results
y_pred_linear_test = start_f.predict(X_test)
r2_score_linear_test = r2_score(y_test, y_pred_linear_test)
r2_score_linear_test



from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


print('R2 score(test): ', r2_score_linear_test)
print('Mean squared error(test): ', mean_squared_error(y_test, y_pred))
print('Root Mean squared error(test): ', np.sqrt(mean_squared_error(y_test, y_pred)))

print('R2 score(train): ', r2_score_linear_train)
print('Mean squared error(train): ', mean_squared_error(y_train, y_pred1))
print('Root Mean squared error(train): ', np.sqrt(mean_squared_error(y_train, y_pred1)))



import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)


import pickle
pickle.dump(start_f,open('allways_multilinear_reg.pkl','wb'))

train_new_1.to_csv("allways_MLR_CSV.csv", index=False)

model = pickle.load(open('allways_multilinear_reg.pkl','rb'))
print(model.predict([[0,0.1,0.11,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))

