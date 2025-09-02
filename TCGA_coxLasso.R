library(correctedC)
library(survminer)
library(dplyr)
library(tableone)
library(reticulate)
library(survivalmodels)
library(survival)
library(devtools)
library(glmnet)
library(survival)
library(fastDummies)
library(readr)
#source('C:/Users/mde4023/downloads/StackedSurvivalData/TCGA/CreateData.R')
setwd('C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA')
trainData=read.csv('trainData.csv')
table(trainData$status)/nrow(trainData)
median(trainData$time)
#nrow(trainData)+nrow(testData)
testData=read.csv('testData.csv')
test_df=testData
trainData_clean <- trainData[, !names(trainData) %in% "X"]

#create the stacking
CreateStack=function(trainData,nrep,myseed){
  set.seed(myseed)
  stacked_df <- do.call(rbind, replicate(nrep, trainData, simplify = FALSE))
  replicated_df<-rbind(stacked_df,trainData)
  maxTime<-max(trainData$time[trainData$status==1])
  timeCens<-runif(nrow(stacked_df),0,maxTime)
  timeComp<-timeCens<stacked_df$time
  stacked_df$time[timeComp]<-timeCens[timeComp]
  stacked_df$status<-ifelse(timeComp,
                           rep(0,nrow(stacked_df)),
                           stacked_df$status)
  stacked_df<-rbind(stacked_df,trainData)
  return(list(
    replicated_df=replicated_df,
    stacked_df=stacked_df,
    trainData=trainData
  ))
}

##################try a lasso cox model######################### seedNum=123 nrep=10
simFunction<-function(seedNum,nrep){
  set.seed(seedNum)
  DataSeg_n=CreateStack(trainData_clean,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(trainData_clean,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  
  ###model 1###
  x <- as.matrix(DataSeg_n$trainData[, !(names(DataSeg_n$trainData) %in% c("time", "status", "X"))])
  y <- with(DataSeg_n$trainData, Surv(time, status))
  
  # Fit LASSO-penalized Cox model
  set.seed(seedNum)
  cox_lasso <- cv.glmnet(
    x = x,
    y = y,
    family = "cox",
    alpha = 1,           # LASSO penalty
    nfolds = 5,          # 5-fold CV to tune lambda
    standardize = TRUE   # standardization recommended
  )
  
  # Extract best model
  best_lambda <- cox_lasso$lambda.min
  lasso_model1 <- glmnet(x, y, family = "cox", alpha = 1, lambda = best_lambda)
  
  ###model 2###
  x <- as.matrix(DataSeg_n$replicated_df[, !(names(DataSeg_n$replicated_df) %in% c("time", "status", "X"))])
  y <- with(DataSeg_n$replicated_df, Surv(time, status))
  
  # Fit LASSO-penalized Cox model
  set.seed(seedNum)
  cox_lasso <- cv.glmnet(
    x = x,
    y = y,
    family = "cox",
    alpha = 1,           # LASSO penalty
    nfolds = 5,          # 5-fold CV to tune lambda
    standardize = TRUE   # standardization recommended
  )
  
  # Extract best model
  best_lambda <- cox_lasso$lambda.min
  lasso_model2 <- glmnet(x, y, family = "cox", alpha = 1, lambda = best_lambda)
  
  ###model 3###
  x <- as.matrix(DataSeg_n$stacked_df[, !(names(DataSeg_n$stacked_df) %in% c("time", "status", "X"))])
  y <- with(DataSeg_n$stacked_df, Surv(time, status))
  
  # Fit LASSO-penalized Cox model
  set.seed(seedNum)
  cox_lasso <- cv.glmnet(
    x = x,
    y = y,
    family = "cox",
    alpha = 1,           # LASSO penalty
    nfolds = 5,          # 5-fold CV to tune lambda
    standardize = TRUE   # standardization recommended
  )
  
  # Extract best model
  best_lambda <- cox_lasso$lambda.min
  lasso_model3<- glmnet(x, y, family = "cox", alpha = 1, lambda = 0.02)
  
  ###model 4###
  x <- as.matrix(DataSeg_1$replicated_df[, !(names(DataSeg_1$replicated_df) %in% c("time", "status", "X"))])
  y <- with(DataSeg_1$replicated_df, Surv(time, status))
  
  # Fit LASSO-penalized Cox model
  set.seed(seedNum)
  cox_lasso <- cv.glmnet(
    x = x,
    y = y,
    family = "cox",
    alpha = 1,           # LASSO penalty
    nfolds = 5,          # 5-fold CV to tune lambda
    standardize = TRUE   # standardization recommended
  )
  
  # Extract best model
  best_lambda <- cox_lasso$lambda.min
  lasso_model4<- glmnet(x, y, family = "cox", alpha = 1, lambda = best_lambda)
  
  ###model 5###
  x <- as.matrix(DataSeg_1$stacked_df[, !(names(DataSeg_1$stacked_df) %in% c("time", "status", "X"))])
  y <- with(DataSeg_1$stacked_df, Surv(time, status))
  
  # Fit LASSO-penalized Cox model
  set.seed(seedNum)
  cox_lasso <- cv.glmnet(
    x = x,
    y = y,
    family = "cox",
    alpha = 1,           # LASSO penalty
    nfolds = 5,          # 5-fold CV to tune lambda
    standardize = TRUE   # standardization recommended
  )
  
  # Extract best model
  best_lambda <- cox_lasso$lambda.min
  lasso_model5<- glmnet(x, y, family = "cox", alpha = 1, lambda = best_lambda)
  
  # Calculate the c index
  testx <- as.matrix(test_df[, colnames(x)]) 
  rfPred1 <- predict(lasso_model1, newx = testx, type = "link")[,1]
  rfPred2 <- predict(lasso_model2, newx = testx, type = "link")[,1]
  rfPred3 <- predict(lasso_model3, newx = testx, type = "link")[,1]
  rfPred4 <- predict(lasso_model4, newx = testx, type = "link")[,1]
  rfPred5 <- predict(lasso_model5, newx = testx, type = "link")[,1]
  
  dnnC1<-UnoC(test_df$time,test_df$status,rfPred1)
  dnnC2<-UnoC(test_df$time,test_df$status,rfPred2)
  dnnC3<-UnoC(test_df$time,test_df$status,rfPred3)
  dnnC4<-UnoC(test_df$time,test_df$status,rfPred4)
  dnnC5<-UnoC(test_df$time,test_df$status,rfPred5)
  
  r=c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5)
  write.csv(t(r),file=paste0('C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA/ResultcoxLasso/NN_',seedNum,'.csv'))
  return(r)
}
simFunction(123,5)

# Set up parallel plan
library(future.apply)
library(reticulate)
ncores <- 30
plan(multisession, workers = ncores)


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

##check if everything is fitted and calculate result
# Directory path
path <- "C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA/ResultcoxLasso/"

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
round(colMeans(all_data[,3:7]),3)
