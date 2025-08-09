# library(devtools)
# remotes::install_github("YushuShi/correctedC")

library(correctedC)
library(survminer)
library(dplyr)
library(tableone)
library(reticulate)
library(survivalmodels)
library(survival)
library(devtools)
library(fastDummies)
library(readr)
use_condaenv("C:/Users/mde4023/AppData/Local/anaconda3/envs/tensorflow", required = TRUE)

setwd('C:/Users/mde4023/documents/github/StackedSurvivalData/TCGA/CreateData.R')
trainData=read.csv('trainData.csv')[,-1]
testData=read.csv('testData.csv')[,-1]

# Extract x, time, and status i=1
x <- as.data.frame(trainData[, !(names(trainData) %in% c("time", "status"))])
y <- Surv(trainData$time, trainData$status)

#function to stack the data
CreateStack=function(data,nrep){
  ntest<-floor(nrow(data)/10)
  trainIndex <- sample(1:nrow(data), nrow(data)-ntest)
  testIndex <- setdiff(1:nrow(data), trainIndex)
  trainData <- data[trainIndex,]
  testData <- data[testIndex,]
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
    stacked_df=stacked_df,
    replicated_df=replicated_df,
    testData=testData,
    trainData=trainData
  ))
}

#########################################
####neural networks#####################
######################################## nrep=1

#fit the stacking
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

##do the replication nrep=10 seedNum=123
simFunction<-function(seedNum,nrep){
  DataSeg_n=CreateStack(trainData,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(trainData,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel1 <-deepsurv(data=DataSeg_n$trainData,
                       time_variable = "time",
                       status_variable = "status",
                       frac = 0.6,
                       activation = "relu",
                       num_nodes = c(32,  64, 128, 256),
                       dropout = 0.1,
                       learning_rate = 0.001,
                       early_stopping = TRUE,
                       epochs = 200,
                       patience = 30,
                       batch_norm = TRUE,
                       batch_size = 128L,
                       shuffle = TRUE)
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel2 <-deepsurv(data=DataSeg_n$replicated_df,
                       time_variable = "time",
                       status_variable = "status",
                       frac = 0.6,
                       activation = "relu",
                       num_nodes = c(32,  64, 128, 256),
                       dropout = 0.1,
                       learning_rate = 0.001,
                       early_stopping = TRUE,
                       epochs = 200,
                       patience = 30,
                       batch_norm = TRUE,
                       batch_size = 128L,
                       shuffle = TRUE)
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel3 <-deepsurv(data=DataSeg_n$stacked_df,
                       time_variable = "time",
                       status_variable = "status",
                       frac = 0.6,
                       activation = "relu",
                       num_nodes = c(32,  64, 128, 256),
                       dropout = 0.1,
                       learning_rate = 0.001,
                       early_stopping = TRUE,
                       epochs = 200,
                       patience = 30,
                       batch_norm = TRUE,
                       batch_size = 128L,
                       shuffle = TRUE)
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel4 <-deepsurv(data=DataSeg_1$replicated_df,
                       time_variable = "time",
                       status_variable = "status",
                       frac = 0.6,
                       activation = "relu",
                       num_nodes = c(32,  64, 128, 256),
                       dropout = 0.1,
                       learning_rate = 0.001,
                       early_stopping = TRUE,
                       epochs = 200,
                       patience = 30,
                       batch_norm = TRUE,
                       batch_size = 128L,
                       shuffle = TRUE)
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel5 <-deepsurv(data=DataSeg_1$stacked_df,
                       time_variable = "time",
                       status_variable = "status",
                       frac = 0.6,
                       activation = "relu",
                       num_nodes = c(32,  64, 128, 256),
                       dropout = 0.1,
                       learning_rate = 0.001,
                       early_stopping = TRUE,
                       epochs = 200,
                       patience = 30,
                       batch_norm = TRUE,
                       batch_size = 128L,
                       shuffle = TRUE)
  dnnPred1 <- predict(dnnModel1, type="risk",newdata = testData)
  dnnPred2 <- predict(dnnModel2, type="risk",newdata = testData)
  dnnPred3 <- predict(dnnModel3, type="risk",newdata = testData)
  dnnPred4 <- predict(dnnModel4, type="risk",newdata = testData)
  dnnPred5 <- predict(dnnModel5, type="risk",newdata = testData)
  
  dnnC1<-UnoC(testData$time,testData$status,dnnPred1)
  dnnC2<-UnoC(testData$time,testData$status,dnnPred2)
  dnnC3<-UnoC(testData$time,testData$status,dnnPred3)
  dnnC4<-UnoC(testData$time,testData$status,dnnPred4)
  dnnC5<-UnoC(testData$time,testData$status,dnnPred5)
  
  return(c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5))
}
round(simFunction(seedNum=777,nrep=10),3)

