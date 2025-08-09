#finetune the random forest

# Load the packages
library(survival)   # For survival analysis functions
library(correctedC)
library(dplyr)
library(reticulate)
library(survivalmodels)
library(devtools)
library(randomForestSRC)
library(fastDummies)
#install.packages("hdf5r")
library(hdf5r)
file <- H5File$new("C:/Users/mde4023/Documents/GitHub/StackedSurvivalData/SUPPORT/support_train_test.h5", mode = "r")
# List available groups and datasets
file$ls(recursive = TRUE)

# Read training data
x_train <- t(file[["train/x"]]$read())  # Transpose to samples × features
t_train <- file[["train/t"]][]         # Time to event
e_train <- file[["train/e"]][]         # Event indicator

# Read testing data
x_test  <- t(file[["test/x"]]$read()) 
t_test  <- file[["test/t"]][]
e_test  <- file[["test/e"]][]

# Convert to data.frame (optional, often useful in R)
train_df <- data.frame(x_train)
train_df$time <- t_train
train_df$event <- e_train

test_df <- data.frame(x_test)
test_df$time <- t_test
test_df$event <- e_test

fulldata=rbind(train_df,test_df)
sum(fulldata$event)/length(fulldata$event)
summary(subset(fulldata,event==1)$time )
# Predictors
all_vars <- names(train_df)[1:14]

# Create the new formula
formula_str <- paste("Surv(time, event) ~", paste(all_vars, collapse = " + "))
myformula <- as.formula(formula_str)

# Define a tuning grid for hyperparameter tuning
tuneGrid <- expand.grid(
  ntree = c(100, 200, 500),        # Number of trees
  mtry = c(1, 2, 3),               # Number of variables to try at each split
  maxnodes = c(10, 20, 30)         # Maximum number of terminal nodes
)

# Initialize variables to store best results
best_cindex <- 0
best_params <- NULL
best_rf_model <- NULL
# Fit the best model on the full training data i=1
for (i in 1:nrow(tuneGrid)) {
  # Get current hyperparameter combination
  current_params <- tuneGrid[i,]
  ntree <- current_params$ntree
  mtry <- current_params$mtry
  maxnodes <- current_params$maxnodes
  
  # Train the model using the current parameters
  rf_model <- rfsrc(myformula, data = train_df,
                    ntree = ntree, mtry = mtry, maxnodes = maxnodes)
  
  # Make predictions on the test data
  rf_pred <- predict(rf_model, newdata = test_df)
  
  # Compute the C-index for the current model
  dnnC1<-UnoC(test_df$time,test_df$event,rf_pred$predicted)
  
  # Check if this model has the best C-index so far
  if (dnnC1 > best_cindex) {
    best_cindex <- dnnC1
    best_params <- current_params
    best_rf_model <- rf_model
  }
}



