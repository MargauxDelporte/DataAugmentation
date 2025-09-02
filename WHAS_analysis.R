# Load the packages
library(smoothHR)      # For the WHAS dataset
library(survival)   # For survival analysis functions
library(ggplot2)    # For plotting
library(ggfortify)  # For autoplotting survival curves
library(correctedC)
library(survminer)
library(dplyr)
library(tableone)
library(reticulate)
library(survivalmodels)
library(intsurv)
library(survival)
library(devtools)
library(keras)
library(randomForestSRC)
library(survivalContour)
library(fastDummies)
library(devtools)
# remotes::install_github("YushuShi/correctedC")

use_condaenv("C:/Users/mde4023/AppData/Local/anaconda3/envs/r-tf", required = TRUE)
#install.packages("hdf5r")
library(hdf5r)
file <- H5File$new("C:/Users/mde4023/Downloads/StackedSurvivalData/WHAS/whas_train_test.h5", mode = "r")
#file <- H5File$new("C:/Users/mde4023/Documents/GitHub/StackedSurvivalData/WHAS/whas_train_test.h5", mode = "r")

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
all_vars <- names(train_df)[1:6]

# Create the new formula
formula_str <- paste("Surv(time, event) ~", paste(all_vars, collapse = " + "))
myformula <- as.formula(formula_str)


#create the stacking
CreateStack=function(trainData,nrep,myseed){
  set.seed(myseed)
  stacked_df <- do.call(rbind, replicate(nrep, trainData, simplify = FALSE))
  replicated_df<-rbind(stacked_df,trainData)
  maxTime<-max(trainData$time[trainData$event==1])
  timeCens<-runif(nrow(stacked_df),0,maxTime)
  timeComp<-timeCens<stacked_df$time
  stacked_df$time[timeComp]<-timeCens[timeComp]
  stacked_df$event<-ifelse(timeComp,
                           rep(0,nrow(stacked_df)),
                           stacked_df$event)
  stacked_df<-rbind(stacked_df,trainData)
  return(list(
    replicated_df=replicated_df,
    stacked_df=stacked_df,
    trainData=trainData
  ))
}
#analysis
#{"learning_rate": 0.023094096518941305, "dropout": 0.017243652343750002, "lr_decay": 0.0009819482421875,
#"momentum": 0.926554443359375, "L2_reg": 2.364680908203125, "batch_norm": false,
#"standardize": true, "n_in": 6, "hidden_layers_sizes": [26, 26, 26], "activation": "selu"}
simFunction<-function(seedNum,nrep){
  DataSeg_n=CreateStack(train_df,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(train_df,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel1 <-deepsurv(data=DataSeg_n$trainData,
                       time_variable = "time",
                       status_variable = "event",
                       learning_rate=0.023094096518941305,
                       lr_decay= 0.0009819482421875,
                       momentum=0.92655444335937,
                       #L2_reg= 2.364680908203125,
                       frac=0.5,    
                       activation="selu",    
                       num_nodes=c(26, 26, 26), 
                       dropout=.017243652343750002,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=50L,
                       batch_norm = F,
                       batch_size=250L,
                       shuffle=TRUE)
  dnnModel2 <-deepsurv(data=DataSeg_n$replicated_df,
                       time_variable = "time",
                       status_variable = "event",
                       learning_rate=0.023094096518941305,
                       lr_decay= 0.0009819482421875,
                       momentum=0.92655444335937,
                       #L2_reg= 2.364680908203125,
                       frac=0.5,    
                       activation="selu",    
                       num_nodes=c(26, 26, 26), 
                       dropout=.017243652343750002,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=50L,
                       batch_norm = F,
                       batch_size=250L,
                       shuffle=TRUE)
  dnnModel3 <-deepsurv(data=DataSeg_n$stacked_df,
                       time_variable = "time",
                       status_variable = "event",
                       learning_rate=0.023094096518941305,
                       lr_decay= 0.0009819482421875,
                       momentum=0.92655444335937,
                       #L2_reg= 2.364680908203125,
                       frac=0.5,    
                       activation="selu",    
                       num_nodes=c(26, 26, 26), 
                       dropout=.017243652343750002,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=50L,
                       batch_norm = F,
                       batch_size=250L,
                       shuffle=TRUE)
  dnnModel4 <-deepsurv(data=DataSeg_1$replicated_df,
                       time_variable = "time",
                       status_variable = "event",
                       learning_rate=0.023094096518941305,
                       lr_decay= 0.0009819482421875,
                       momentum=0.92655444335937,
                       #L2_reg= 2.364680908203125,
                       frac=0.5,    
                       activation="selu",    
                       num_nodes=c(26, 26, 26), 
                       dropout=.017243652343750002,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=50L,
                       batch_norm = F,
                       batch_size=250L,
                       shuffle=TRUE)
  dnnModel5 <-deepsurv(data=DataSeg_1$stacked_df,
                       time_variable = "time",
                       status_variable = "event",
                       learning_rate=0.023094096518941305,
                       lr_decay= 0.0009819482421875,
                       momentum=0.92655444335937,
                       #L2_reg= 2.364680908203125,
                       frac=0.5,    
                       activation="selu",    
                       num_nodes=c(26, 26, 26), 
                       dropout=.017243652343750002,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=50L,
                       batch_norm = F,
                       batch_size=250L,
                       shuffle=TRUE)
  
  dnnPred1 <- predict(dnnModel1, type="risk",newdata = test_df)
  dnnPred2 <- predict(dnnModel2, type="risk",newdata = test_df)
  dnnPred3 <- predict(dnnModel3, type="risk",newdata = test_df)
  dnnPred4 <- predict(dnnModel4, type="risk",newdata = test_df)
  dnnPred5 <- predict(dnnModel5, type="risk",newdata = test_df)
  
  dnnC1<-UnoC(test_df$time,test_df$event,dnnPred1)
  dnnC2<-UnoC(test_df$time,test_df$event,dnnPred2)
  dnnC3<-UnoC(test_df$time,test_df$event,dnnPred3)
  dnnC4<-UnoC(test_df$time,test_df$event,dnnPred4)
  dnnC5<-UnoC(test_df$time,test_df$event,dnnPred5)
  
  coxmd=coxph(myformula,data=DataSeg_n$trainData)
  coxmdPred1 <- predict(coxmd, type="risk",newdata = test_df)
  coxmdC1<-UnoC(test_df$time,test_df$event,coxmdPred1)
  r=c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5,coxmdC1)
  write.csv(t(r),file=paste0('C:/Users/mde4023/Downloads/StackedSurvivalData/WHAS/resultNN/NN_',seedNum,'.csv'))
  return(c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5,coxmdC1))
}


#round(simFunction(123,5),3)
#sum/test_df$event
#[1]  123.0000000   0.8373587   0.8577635   0.8527725   0.8604064   0.8460137   0.8252750
# Set up cluster
n_cores <- 25
library(future.apply)
library(reticulate)

# Set up parallel plan
ncores <- parallel::detectCores() - 1
plan(multisession, workers = ncores)

# Make sure Python env is used inside each worker
use_python_env <- function() {
  reticulate::use_condaenv("r-tf", required = TRUE)
}

# Wrap simFunction to ensure Python env is initialized in each worker
simFunction_parallel <- function(seedNum, nrep) {
  use_python_env()   # ensure pycox etc. is available
  simFunction(seedNum = seedNum, nrep = nrep)
}

# Seeds to run
seeds <- 1:999
nrep <- 5

# Run in parallel with reproducible RNG
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep)
}, future.seed = TRUE)

# Reset future plan
plan(sequential)

# Directory path
path <- "C:/Users/mde4023/Downloads/StackedSurvivalData/WHAS/resultNN/"

# List all files
files <- list.files(path, full.names = FALSE)

# Extract the numeric part after "RF_"
nums <- as.numeric(sub(".*NN_([0-9]+).*", "\\1", files))

# Combine file names with extracted numbers
result <- data.frame(file = files, value = nums)

todo=setdiff(1:999,result$value)
todo
length(todo)
# Set up parallel plan
n_cores <- min(length(todo),30)
plan(multisession, workers = 30)

# Make sure Python env is used inside each worker
use_python_env <- function() {
  reticulate::use_condaenv("r-tf", required = TRUE)
}

# Wrap simFunction to ensure Python env is initialized in each worker
simFunction_parallel <- function(seedNum, nrep) {
  use_python_env()   # ensure pycox etc. is available
  simFunction(seedNum = seedNum, nrep = nrep)
}

# Seeds to run
seeds <- todo
nrep <- 5

# Run in parallel with reproducible RNG
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep)
}, future.seed = TRUE)

# Reset future plan
plan(sequential)

warnings()
#> round(colMeans(results_matrix),3)

##check if everything is fitted and calculate result
# Directory path
path <- "C:/Users/mde4023/Downloads/StackedSurvivalData/WHAS/resultNN/"

# List all files
files <- list.files(path, full.names = FALSE)

# Extract the numeric part after "NN_"
nums <- as.numeric(sub(".*NN_([0-9]+).*", "\\1", files))

# Combine file names with extracted numbers
result <- data.frame(file = files, value = nums)

todo=setdiff(1:999,result$value)
todo
# Read and rbind all files
files <- list.files(path, full.names = TRUE)
all_data <- lapply(files, read.csv) %>% bind_rows()

# If you want to keep track of which file each row came from
all_data <- lapply(files, function(f) {
  df <- read.csv(f)
  df$file_source <- basename(f)
  df
}) %>% bind_rows()
head(all_data)
round(colMeans(all_data[,3:8]),3)

##################try a random forest#########################
simFunction<-function(seedNum,nrep){
  DataSeg_n=CreateStack(train_df,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(train_df,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  rfModel1 <- rfsrc(myformula,DataSeg_n$trainData,ntree = 500, mtry = 2, maxnodes = 10)
  rfModel2 <- rfsrc(myformula,DataSeg_n$replicated_df,ntree = 500, mtry = 2, maxnodes = 10)
  rfModel3 <- rfsrc(myformula,DataSeg_n$stacked_df,ntree = 500, mtry = 2, maxnodes = 10)
  rfModel4 <- rfsrc(myformula,DataSeg_1$replicated_df,ntree = 500, mtry = 2, maxnodes = 10)
  rfModel5 <- rfsrc(myformula,DataSeg_1$stacked_df,ntree = 500, mtry = 2, maxnodes = 10)
  
  rfPred1 <- predict(rfModel1,newdata = test_df)
  rfPred2 <- predict(rfModel2,newdata = test_df)
  rfPred3 <- predict(rfModel3,newdata = test_df)
  rfPred4 <- predict(rfModel4,newdata = test_df)
  rfPred5 <- predict(rfModel5,newdata = test_df)
  
  dnnC1<-UnoC(test_df$time,test_df$event,rfPred1$predicted)
  dnnC2<-UnoC(test_df$time,test_df$event,rfPred2$predicted)
  dnnC3<-UnoC(test_df$time,test_df$event,rfPred3$predicted)
  dnnC4<-UnoC(test_df$time,test_df$event,rfPred4$predicted)
  dnnC5<-UnoC(test_df$time,test_df$event,rfPred5$predicted)
  
  r=c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5)
  write.csv(t(r),file=paste0('C:/Users/mde4023/Downloads/StackedSurvivalData/WHAS/resultRF/RF_',seedNum,'.csv'))
  return(c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5))
}
round(simFunction(123,10),3)
#[1] 123.000   0.876   0.907   0.909   0.886   0.884

# Set up parallel plan
library(future.apply)
library(reticulate)
plan(multisession, workers = 25)


# Wrap simFunction to ensure Python env is initialized in each worker
simFunction_parallel <- function(seedNum, nrep) {
  simFunction(seedNum = seedNum, nrep = 5)
}

# Seeds to run
seeds <- 1:999
nrep <- 5

# Run in parallel with reproducible RNG
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep)
}, future.seed = TRUE)

# Reset future plan
plan(sequential)

##check if everything is fitted and calculate result
# Directory path
path <- "C:/Users/mde4023/Downloads/StackedSurvivalData/WHAS/resultRF/"

# List all files
files <- list.files(path, full.names = FALSE)

# Extract the numeric part after "NN_"
nums <- as.numeric(sub(".*RF_([0-9]+).*", "\\1", files))

# Combine file names with extracted numbers
result <- data.frame(file = files, value = nums)

todo=setdiff(1:999,result$value)
todo
# Read and rbind all files
files <- list.files(path, full.names = TRUE)
all_data <- lapply(files, read.csv) %>% bind_rows()

# If you want to keep track of which file each row came from
all_data <- lapply(files, function(f) {
  df <- read.csv(f)
  df$file_source <- basename(f)
  df
}) %>% bind_rows()
head(all_data)
round(colMeans(all_data[,3:7]),3)
