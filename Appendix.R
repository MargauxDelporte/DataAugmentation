#stacking simulation
# Load necessary libraries
library(randomForestSRC)
library(survival)
library(pROC)
library(devtools)
library(future.apply)
library(reticulate)
library(dplyr)

#source('~/Stacking/correctedC.R')
#remotes::install_github("YushuShi/correctedC")
library(correctedC)
# Set seed for reproducibility
set.seed(123)

# Simulate data with nonlinear predictors
n <- 750  # number of samples
p <- 8    # number of predictors

# Simulate nonlinear predictors
X <- matrix(rnorm(n * p), ncol = p)
colnames(X) <- paste0("X", 1:p)


# Simulate complex nonlinear effect for the linear predictor
coef=rnorm(n=25,mean=0,sd=1)
linpred <- (
  2.0 * (X[,1]) +
    2.0 * (abs(X[,2]) + 1) -
    2.5 * (X[,3])^2 +
    2.0 * X[,4] * X[,5] +
    2.5 * X[,5] * X[,6] -
    2.0 * X[,7] * X[,8] 
)

# Mild noise
linpred <- linpred + rnorm(n, 0, 0.1)

# Convert to survival times via exponential distribution with rate based on linpred
baseline_hazard <- 0.1
event_time <- rexp(n, rate = baseline_hazard * exp(linpred))  # nonlinearly varying hazard

# Simulate independent censoring times
censor_time <- rexp(n, rate = 0.05)  # less frequent censoring

# Observed time and event indicator
time <- pmin(event_time, censor_time)
event <- as.numeric(event_time <= censor_time)

# Create data frame
sim_data <- data.frame(time = time, event = event, X)
traindata=sim_data[1:floor(0.8*n),]
testdata=sim_data[(floor(0.8*n)+1):n,]
myformula=as.formula('Surv(time, event) ~ .')
train_df=traindata
test_df=testdata
#create the stacking trainData=train_df2
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

#random forest on data with a c index of 0.8
train_df=traindata
test_df=testdata
myformula=as.formula('Surv(time, event) ~ .')
simFunction<-function(seedNum,nrep){
  DataSeg_n=CreateStack(train_df,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(train_df,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set.seed(seedNum)
  
  #best parameters based on fine tuning
  ntree=500
  rfModel1 <- rfsrc(myformula,DataSeg_n$trainData,ntree = ntree)
  rfModel2 <- rfsrc(myformula,DataSeg_n$replicated_df,ntree = ntree)
  rfModel3 <- rfsrc(myformula,DataSeg_n$stacked_df,ntree = ntree)
  rfModel4 <- rfsrc(myformula,DataSeg_1$replicated_df,ntree = ntree)
  rfModel5 <- rfsrc(myformula,DataSeg_1$stacked_df,ntree = ntree)
  
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
  write.csv(t(r),file=paste0('~/RF1/RF_',seedNum,'.csv'))
  return(c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5))
}

parallel::detectCores()
n_cores <- 20
library(future.apply)
library(reticulate)

# Set up parallel plan
ncores <- parallel::detectCores() - 1
plan(multisession, workers = 3)

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

train_df2=traindata
#system.time({
#  r=round(simFunctionRF(123,5),3)
#})
#[1]123.000   0.811   0.842   0.839   0.825   0.828
#   user  system elapsed 
#36.389   0.653  37.151 
#parallel::detectCores()
#n_cores <- 50
#library(future.apply)
#library(reticulate)

# Set up parallel plan
#ncores <- parallel::detectCores() - 1
#plan(multisession, workers = 3)

# Wrap simFunction to ensure Python env is initialized in each worker
#simFunction_parallel <- function(seedNum, nrep) {
#  simFunction(seedNum = seedNum, nrep = nrep)
#}

# Seeds to run
#seeds <- 1:999
#nrep <- 5

# Run in parallel with reproducible RNG
#results_matrix <- future_sapply(seeds, function(s) {
#  simFunction_parallel(seedNum = s, nrep = nrep)
#}, future.seed = TRUE)

# Reset future plan
#plan(sequential)



simFunctionRF<-function(seedNum,nrep,mypath,npermute){
  train_df2=traindata
  
  #permute some data
  train_df2$time[1:floor(npermute*600)]=sample(train_df2$time,replace=F)[1:floor(npermute*600)]
  train_df2$event[1:floor(npermute*600)]=sample(train_df2$event,replace=F)[1:floor(npermute*600)]
  
  #create stacks
  DataSeg_n=CreateStack(train_df2,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(train_df2,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set.seed(seedNum)
  
  #best parameters based on fine tuning
  ntree=500
  rfModel1 <- rfsrc(myformula,DataSeg_n$trainData,ntree = ntree)
  rfModel2 <- rfsrc(myformula,DataSeg_n$replicated_df,ntree = ntree)
  rfModel3 <- rfsrc(myformula,DataSeg_n$stacked_df,ntree = ntree)
  rfModel4 <- rfsrc(myformula,DataSeg_1$replicated_df,ntree = ntree)
  rfModel5 <- rfsrc(myformula,DataSeg_1$stacked_df,ntree = ntree)
  
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
  r=c(seedNum,npermute,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5)
  write.csv(t(r),file=paste0(mypath,'RF_',npermute*10,seedNum,'.csv'))
  return(c(npermute,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5))
}
system.time({
  r=round(simFunctionRF(10,5,mypath,npermute=.20),3)
})
#user  system elapsed 
#96.24    1.79   10.65 

#parallel::detectCores()
library(future.apply)
library(reticulate)
library(dplyr)
# Set up parallel plan
ncores <- 25
plan(multisession, workers = ncores)
path='C:/Users/mde4023/Downloads/StackedSurvivalData/SimulatedForest/Permutation_result/90/'
# Wrap simFunction to ensure Python env is initialized in each worker
simFunction_parallel <- function(seedNum,nrep,mypath,npermute) {
  simFunctionRF(seedNum = seedNum, nrep = nrep,mypath=mypath,npermute=npermute)
}

# Seeds to run
seeds <- 1:999
nrep <- 5
#path='C:/Users/mde4023/Downloads/StackedSurvivalData/SimulatedForest/Permutation_result/'
#setwd(path)

##check if everything is fitted and calculate result
# Directory path
path2 <- path

# List all files
files <- list.files(path2, full.names = FALSE)

# Extract the numeric part after "NN_"
nums <- as.numeric(sub(".*RF_9([0-9]+).*", "\\1", files))

# Combine file names with extracted numbers
result <- data.frame(file = files, value = nums)

todo=setdiff(1:999,result$value)
todo
length(todo)
# Seeds to run
seeds <- todo

# Run in parallel with reproducible RNG
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep,mypath=path,npermute=0.9)
}, future.seed = TRUE)#

# Reset future plan
plan(sequential)

##check if everything is fitted and calculate result
# Directory path

# List all files
files <- list.files(path, full.names = FALSE)

# Extract the numeric part after "NN_"
nums <- as.numeric(sub(".*RF_9([0-9]+).*", "\\1", files))

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


#result=round(simFunctionRF(123,10,0.1),3)
#for(i in seq(from=0.2,to=1,by=0.1)){
#  result=rbind(result,round(simFunctionRF(123,10,i),3))
#}
#result[,7]=rep(0,10)
#result2=cbind(result,as.vector(result[,4]-result[,2]))


#whas permutation approach
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

# Create the new formula npermute=0.5 seedNum=123 nrep=10
formula_str <- paste("Surv(time, event) ~", paste(all_vars, collapse = " + "))
myformula <- as.formula(formula_str)
n=nrow(train_df)
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
simFunctionRF<-function(seedNum,nrep,npermute,mypath=path){
  train_df2=train_df
  
  #permute some data
  train_df2$time[1:floor(npermute*n)]=sample(train_df2$time,replace=F)[1:floor(npermute*n)]
  train_df2$event[1:floor(npermute*n)]=sample(train_df2$event,replace=F)[1:floor(npermute*n)]
  
  DataSeg_n=CreateStack(train_df2,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(train_df2,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set.seed(seedNum)
  
  #best parameters based on fine tuning

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
  write.csv(t(r),file=paste0(mypath,npermute*100,'/RF_',npermute*10,seedNum,'.csv'))
  return(r)
}
result=round(simFunctionRF(123,5,0.1),3)
for(i in seq(from=0.2,to=1,by=0.1)){
  result=rbind(result,round(simFunctionRF(123,5,i),3))
}
result2=cbind(result,as.vector(result[,4]-result[,2]))


ncores <- 25
plan(multisession, workers = ncores)
# Wrap simFunction to ensure Python env is initialized in each worker
simFunction_parallel <- function(seedNum,nrep,mypath,npermute) {
  simFunctionRF(seedNum = seedNum, nrep = nrep,mypath=path,npermute=npermute)
}
seeds <- 1:999
nrep <- 5
path='C:/Users/mde4023/Downloads/StackedSurvivalData/SimulatedForest/Permutation_result/WHAS/'
##simFunctionRF(seedNum = 123, nrep = 5,mypath=path,npermute=0.9)
##check if everything is fitted and calculate result
#seeds=todo
# Run in parallel with reproducible RNG
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep,mypath=path,npermute=0.1)
}, future.seed = TRUE)
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep,mypath=path,npermute=0.2)
}, future.seed = TRUE)
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep,mypath=path,npermute=0.3)
}, future.seed = TRUE)
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep,mypath=path,npermute=0.4)
}, future.seed = TRUE)
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep,mypath=path,npermute=0.5)
}, future.seed = TRUE)
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep,mypath=path,npermute=0.6)
}, future.seed = TRUE)
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep,mypath=path,npermute=0.7)
}, future.seed = TRUE)
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep,mypath=path,npermute=0.8)
}, future.seed = TRUE)
results_matrix <- future_sapply(seeds, function(s) {
  simFunction_parallel(seedNum = s, nrep = nrep,mypath=path,npermute=0.9)
}, future.seed = TRUE)

# Reset future plan
plan(sequential)

library(dplyr)
path0='C:/Users/mde4023/Downloads/StackedSurvivalData/SimulatedForest/Permutation_result/WHAS/';i=80
for(i in seq(from=10,to=90,by=10)){
  path=paste0(path0,i)
  files <- list.files(paste0(path0,i), full.names = FALSE)
  
  # Extract the numeric part after "NN_"
  nums <- as.numeric(sub(paste0(".*RF_",i/10,"([0-9]+).*"), "\\1", files))
  
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
  #head(all_data)
  print(c(i/10,round(colMeans(all_data[,3:7]),3),round(mean(all_data[,3])-mean(all_data[,4]),3),length(result$value)))
}
