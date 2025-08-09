# Install packages if they are not already installed
#install.packages("smoothHR")
#install.packages("survival")
#install.packages("ggplot2")
#install.packages("ggfortify")
# install_github("YushuShi/correctedC")
# install_github("YushuShi/survivalContour")
# Load the packages
library(smoothHR)      # For the WHAS dataset
library(survival)   # For survival analysis functions
library(ggplot2)    # For plotting
library(ggfortify)  # For autoplotting survival curves
library(devtools)
library(correctedC)
library(survminer)
library(dplyr)
library(tableone)
library(reticulate)
library(survivalmodels)
library(survival)
library(devtools)
library(survivalContour)
library(randomForestSRC)
library(fastDummies)
library(reticulate)
#use_condaenv("/Users/yushushi/opt/anaconda3/envs/tensorflow", required = TRUE)
#py_install(c("pycox", "torchtuples"))
use_condaenv("C:/Users/mde4023/AppData/Local/anaconda3/envs/r-tf", required = TRUE)


data(whas500)
# Check the structure of the dataset
str(whas500)
View(whas500)
ntest<-floor(nrow(whas500)/10)
trainIndex <- sample(1:nrow(whas500), nrow(whas500)-ntest)
testIndex <- setdiff(1:nrow(whas500), trainIndex)
trainData <- whas500[trainIndex,]
testData <- whas500[testIndex,]

names(whas500)
linear_fit <- coxph(
  Surv(whas500$lenfol, whas500$fstat)~age+gender+hr+sysbp+diasbp+bmi+cvd+afb+sho+chf+av3+miord+mitype,
  data=whas500)
linear_fit
dnnModel1 <-deepsurv(data=trainData,
                     time_variable = "lenfol",
                     status_variable = "fstat",
                     frac=0.5,    
                     activation="relu",    
                     num_nodes=c(64L,128L,64L), 
                     dropout=0.2,          
                     early_stopping=TRUE, 
                     epochs=1000L,
                     patience=50L,
                     batch_norm = TRUE,
                     batch_size=250L,
                     shuffle=TRUE)
dnnPred1 <- predict(dnnModel1, type="risk",newdata = testData)
dnnPred2 <- predict(linear_fit, type="risk",newdata =testData)
CC1<-UnoC(testData$lenfol, testData$fstat, dnnPred1)
CC2<-UnoC(testData$lenfol, testData$fstat, dnnPred2)
