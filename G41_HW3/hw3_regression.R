########
# HW3 
# Instructor: Dr. Thomas Price
#
# Team Members:
# Srujana Rachakonda (srachak)
# Rajshree Jain (rjain27)
#
# @author: Krishna Gadiraju/kgadira, Yang Shi/yangatrue
#
#########

# Write code for regression here
alda_regression <- function(x_train, x_test, y_train, regression_type){
  # Perform regression (linear/lasso)
  
  # Inputs:
    # x_train: training data frame(19 variables, x1-x19)
    # x_test: test data frame(19 variables, x1-x19)
    # y_train: dependent variable, training data (vector, continous type)
    # regression_type: specifies type of regression, string variable, can be of type 'linear' or 'lasso'
  
  # General Information:
    # Instructions for specific regression types:
      # linear: no cross validation
      # lasso: use 10-fold cross validation to determine optimal lambda
  predicted_y = list()
  # Output:
    # A list with two elements, first element = model generated, second element = predictions on test data (vector) 
  # allowed packages: R-base, glmnet
  # Function hints: Read the documentation for the functions glmnet, cv.glmnet, predict
  # Ridge and Lasso regression hints: Lambda is the hyperparameter
  if(regression_type == 'linear'){ 
    # ~ 2-3 lines of code
    # write code for building a linear regression model using x_train, y_train
    # Optional: can you use glmnet to do simple linear regression as well?
    # Explore away!  
    # Hint: Think of what the lambda value means for linear regression without regularization.
    slr_fit <- glmnet(x_train, y_train ,lambda = 0)
    #slr_fit <- lm(y_train~.,data = x_train)
    # predict using the model
    predicted_y <-  predict(slr_fit, newx = x_test)
    
  }else{
    # ~ 2-3 lines of code
    # write code for lasso regression here
    # Use 10-fold cross validation *on the training data only* to tune the hyperparameter lambda
    # using MSE (mean squared error) as the measure
    # Hint: use ?cv.glmnet to read more on how lasso uses CV
    lasso_fit <- cv.glmnet(x_train, y_train, alpha = 1)
    # predict on x_test using the model that gives least MSE
    predicted_y <- predict(lasso_fit, newx = x_test)
    
  }
  return(predicted_y)
}



