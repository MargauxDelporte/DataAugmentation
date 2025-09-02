# Load the packages
library(survival)   # For survival analysis functions
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
library(fastDummies)
use_condaenv("C:/Users/mde4023/AppData/Local/anaconda3/envs/r-tf", required = TRUE)
#install.packages("hdf5r")
library(hdf5r)
file <- H5File$new("C:/Users/mde4023/Downloads/StackedSurvivalData/SUPPORT/support_train_test.h5", mode = "r")
#file <- H5File$new("C:/Users/mde4023/Documents/GitHub/StackedSurvivalData/SUPPORT/support_train_test.h5", mode = "r")
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

seed_R=123

#analysis based on the parameters from the deepsurv paper
#{"learning_rate": 0.03279227024343838, "dropout": 0.1982763671875, "lr_decay": 0.000645986328125, "momentum": 0.9450444335937499, "L2_reg": 3.54434228515625, "batch_norm": false, "standardize": true, "n_in": 14, "hidden_layers_sizes": [33, 33, 33], "activation": "selu"}
#added an extra layer bc more data in 2-5 NN
simFunction<-function(seedNum,nrep){
  DataSeg_n=CreateStack(train_df,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(train_df,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel1 <-deepsurv(data=DataSeg_n$trainData,
                       time_variable = "time",
                       status_variable = "event",
                       learning_rate=0.001503067,
                       lr_decay= 0.002486383,
                       momentum=0.886958,
                       #L2_reg=  1.484671e-06,
                       frac=0.8873561,    
                       activation="relu",    
                       num_nodes=c(128,  64), 
                       dropout= 0.4836538,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=30L,
                       batch_norm = T,
                       batch_size=128,
                       shuffle=TRUE)
  dnnModel2 <-deepsurv(data=DataSeg_n$replicated_df,
                       time_variable = "time",
                       status_variable = "event",
                       learning_rate=0.001503067,
                       lr_decay= 0.002486383,
                       momentum=0.886958,
                       #L2_reg=  1.484671e-06,
                       frac=0.8873561,    
                       activation="relu",    
                       num_nodes=c(128,  64), 
                       dropout= 0.4836538,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=30L,
                       batch_norm = T,
                       batch_size=128,
                       shuffle=TRUE)
  dnnModel3 <-deepsurv(data=DataSeg_n$stacked_df,
                       time_variable = "time",
                       status_variable = "event",
                       learning_rate=0.001503067,
                       lr_decay= 0.002486383,
                       momentum=0.886958,
                       #L2_reg=  1.484671e-06,
                       frac=0.8873561,    
                       activation="relu",    
                       num_nodes=c(128,  64), 
                       dropout= 0.4836538,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=30L,
                       batch_norm = T,
                       batch_size=128,
                       shuffle=TRUE)
  dnnModel4 <-deepsurv(data=DataSeg_1$replicated_df,
                       time_variable = "time",
                       status_variable = "event",
                       learning_rate=0.001503067,
                       lr_decay= 0.002486383,
                       momentum=0.886958,
                       #L2_reg=  1.484671e-06,
                       frac=0.8873561,    
                       activation="relu",    
                       num_nodes=c(128,  64), 
                       dropout= 0.4836538,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=30L,
                       batch_norm = T,
                       batch_size=128,
                       shuffle=TRUE)
  dnnModel5 <-deepsurv(data=DataSeg_1$stacked_df,
                       time_variable = "time",
                       status_variable = "event",
                       learning_rate=0.001503067,
                       lr_decay= 0.002486383,
                       momentum=0.886958,
                       #L2_reg=  1.484671e-06,
                       frac=0.8873561,    
                       activation="relu",    
                       num_nodes=c(128,  64), 
                       dropout= 0.4836538,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=30L,
                       batch_norm = T,
                       batch_size=128,
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
  write.csv(t(r),file=paste0('C:/Users/mde4023/Downloads/StackedSurvivalData/SUPPORT/resultNN/NN_',seedNum,'.csv'))
  return(c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5,coxmdC1))
}
#?write.csv
#system.time({
#  result <- round(simFunction(123, 10), 3)
#})
#print(result)
# user  system elapsed 
#587.01  404.03  667.13 
#> print(result)
#[1] 123.000   0.865   0.727   0.780   0.840   0.818   0.780


# Set up cluster
library(future.apply)
library(reticulate)

# Set up parallel plan
n_cores <- 20
plan(multisession, workers = 20)

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

# Optional: convert to matrix or data.frame
results_matrix <- t(results_matrix)
colnames(results_matrix) <- c("seed", "dnnC1", "dnnC2", "dnnC3", "dnnC4", "dnnC5", "coxC1")
results_matrix
round(colMeans(results_matrix),3)

reticulate::py_last_error()

# Directory path
path <- "C:/Users/mde4023/Downloads/StackedSurvivalData/SUPPORT/resultNN/"

# List all files
files <- list.files(path, full.names = FALSE)

# Extract the numeric part after "NN_"
nums <- as.numeric(sub(".*NN_([0-9]+).*", "\\1", files))

# Combine file names with extracted numbers
result <- data.frame(file = files, value = nums)

todo=setdiff(1:999,result$value)
todo
length(todo)
# Set up parallel plan
n_cores <- min(length(todo),30)
plan(multisession, workers = 1)

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


##check if everything is fitted and calculate result
# Directory path
path <- "C:/Users/mde4023/Downloads/StackedSurvivalData/SUPPORT/resultNN/"

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
# FOR (33 33 33) NODES
#V1      V2      V3      V4      V5      V6      V7 
#500.000   0.712   0.730   0.733   0.726   0.729   0.780 
#0.806 0.688 0.692 0.787 0.785 0.780 
#all: 0.805 0.679 0.690 0.787 0.782 0.780 

##################fit random forest#########################
seedNum=123
nrep=5
library(randomForestSRC)
simFunction<-function(seedNum,nrep){
  DataSeg_n=CreateStack(train_df,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(train_df,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  
  #best parameters based on fine tuning
  ntree <- 200
  mtry <- 1
  maxnodes <- 10
  
  #fit random forests models
  rfModel1 <- rfsrc(myformula,DataSeg_n$trainData,ntree = ntree, mtry = mtry, maxnodes = maxnodes)
  rfModel2 <- rfsrc(myformula,DataSeg_n$replicated_df,ntree = ntree, mtry = mtry, maxnodes = maxnodes)
  rfModel3 <- rfsrc(myformula,DataSeg_n$stacked_df,ntree = ntree, mtry = mtry, maxnodes = maxnodes)
  rfModel4 <- rfsrc(myformula,DataSeg_1$replicated_df,ntree = ntree, mtry = mtry, maxnodes = maxnodes)
  rfModel5 <- rfsrc(myformula,DataSeg_1$stacked_df,ntree = ntree, mtry = mtry, maxnodes = maxnodes)
  
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
  write.csv(t(r),file=paste0('C:/Users/mde4023/Downloads/StackedSurvivalData/SUPPORT/resultRF/RF_',seedNum,'.csv'))
  return(c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5))
}
#reproducable
#round(simFunction(123,10),3)
#round(simFunction(123,10),3)
# Set up parallel plan
library(future.apply)
library(reticulate)
n_cores <- 32
plan(multisession, workers = n_cores)

# Wrap simFunction to ensure Python env is initialized in each worker
simFunction_parallel <- function(seedNum, nrep) {
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
path <- "C:/Users/mde4023/Downloads/StackedSurvivalData/SUPPORT/resultRF/"

# List all files
files <- list.files(path, full.names = FALSE)

# Extract the numeric part after "RF_"
nums <- as.numeric(sub(".*RF_([0-9]+).*", "\\1", files))

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

##check if everything is fitted and calculate result
# Directory path
path <- "C:/Users/mde4023/Downloads/StackedSurvivalData/SUPPORT/resultRF/"

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
round(colMeans(all_data[,2:7]),3)
