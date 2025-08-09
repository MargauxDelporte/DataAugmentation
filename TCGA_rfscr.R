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
library(readr)
#source('C:/Users/mde4023/downloads/StackedSurvivalData/TCGA/CreateData.R')
setwd('C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA')
trainData=read.csv('trainData.csv')
testData=read.csv('testData.csv')
test_df=testData
trainData_clean <- trainData[, !names(trainData) %in% "X"]
rfModel1 <- rfsrc(Surv(time,status)~.,data=trainData_clean)

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

##################try a random forest######################### seedNum=123 nrep=10
simFunctionRF<-function(seedNum,nrep){
  set.seed(seedNum)
  DataSeg_n=CreateStack(trainData_clean,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(trainData_clean,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  rfModel1 <- rfsrc(
    formula = Surv(time, status) ~ .,
    data = DataSeg_n$trainData,
    ntree = 500,
    mtry = 10,
    nodesize = 10,
    block.size = 1,  # prevents full data in memory
    importance = "none",
    forest = TRUE
  )
  rfModel2 <- rfsrc(
    formula = Surv(time, status) ~ .,
    data = DataSeg_n$replicated_df,
    ntree = 500,
    mtry = 10,
    nodesize = 10,
    block.size = 1,  # prevents full data in memory
    importance = "none",
    forest = TRUE
  )
  rfModel3 <- rfsrc(
    formula = Surv(time, status) ~ .,
    data = DataSeg_n$stacked_df,
    ntree = 500,
    mtry = 10,
    nodesize = 10,
    block.size = 1,  # prevents full data in memory
    importance = "none",
    forest = TRUE
  )
  rfModel4 <- rfsrc(
    formula = Surv(time, status) ~ .,
    data = DataSeg_1$replicated_df,
    ntree = 500,
    mtry = 10,
    nodesize = 10,
    block.size = 1,  # prevents full data in memory
    importance = "none",
    forest = TRUE
  )
  rfModel5 <- rfsrc(
    formula = Surv(time, status) ~ .,
    data = DataSeg_1$stacked_df,
    ntree = 500,
    mtry = 10,
    nodesize = 10,
    block.size = 1,  # prevents full data in memory
    importance = "none",
    forest = TRUE
  )

  rfPred1 <- predict(rfModel1,newdata = test_df)
  rfPred2 <- predict(rfModel2,newdata = test_df)
  rfPred3 <- predict(rfModel3,newdata = test_df)
  rfPred4 <- predict(rfModel4,newdata = test_df)
  rfPred5 <- predict(rfModel5,newdata = test_df)
  
  dnnC1<-UnoC(test_df$time,test_df$status,rfPred1$predicted)
  dnnC2<-UnoC(test_df$time,test_df$status,rfPred2$predicted)
  dnnC3<-UnoC(test_df$time,test_df$status,rfPred3$predicted)
  dnnC4<-UnoC(test_df$time,test_df$status,rfPred4$predicted)
  dnnC5<-UnoC(test_df$time,test_df$status,rfPred5$predicted)
  
  r=c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5)
  return(c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5))
}
round(simFunctionRF(777,10),3)
#[1] 777.000   0.637   0.664   0.640   0.638   0.617
