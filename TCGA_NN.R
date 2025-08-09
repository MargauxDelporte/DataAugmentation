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
library(survivalContour)
library(randomForestSRC)
library(fastDummies)
library(readr)
source('C:/Users/mde4023/downloads/StackedSurvivalData/TCGA/CreateData.R')
#use_condaenv("/Users/yushushi/opt/anaconda3/envs/tensorflow", required = TRUE)
#use_condaenv("C:/Users/mde4023/AppData/Local/anaconda3/envs/tensorflow", required = TRUE)
use_condaenv("C:/Users/mde4023/AppData/Local/anaconda3/envs/r-tf", required = TRUE)
data=mydata
#data=trainData
#processing data
length(names(data))
table(data$tumor_tissue_site)
table(data$gender)
table(data$radiation_therapy)
table(data$histological_type)
table(data$residual_tumor)
table(data$ethnicity)
#table(data$DES)
# Creating dummy variables
data$Sex_M <- ifelse(data$gender == "male", 1, 0)
data$radioT_Y <- ifelse(data$radiation_therapy == "yes", 1, 0)
data$r1 <- ifelse(data$residual_tumor == "r1", 1, 0)
data$r2 <- ifelse(data$residual_tumor == "r2", 1, 0)
data$rx <- ifelse(data$residual_tumor == "rx", 1, 0)

sum(is.na(data$gender))
data=data[-which(is.na(data$residual_tumor)),]
data=data[-which(is.na(data$radiation_therapy)),]

# Create the formula
dummy_vars <- c("Sex_M",'radioT_Y','r1','r2','rx')

# Other variables
other_vars <- names(data)[16:1014]

# Combine dummy variables and other predictors
all_vars <- c(dummy_vars, other_vars)

# Create the new formula
#error because expression is to large
#formula_str <- paste("Surv(time, status) ~", paste(all_vars, collapse = " + "))
#
#coxph(Surv(time, status) ~ ., data = data[, c("time", "status", all_vars)])

#only select relevant variables in the data
mydata=data[,c('time', 'status', all_vars)]
#coxph(Surv(time, status) ~ .,data=mydata)

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

##finetune cirrhosis neural network

#do a grid search
result=c()
grid <- expand.grid(
  dropout = c(0.1, 0.2, 0.3, 0.4, 0.5),
  num_nodes = list(
    c(16L, 32L, 64L), c(32L, 64L, 128L, 256L),
    c(16L, 16L, 32L, 32L), c(64L, 128L, 256L, 512L)
  ),
  activation = c("relu", "elu", "tanh", "sigmoid"),
  learning_rate = c(0.001, 0.01, 0.1),
  Patience=c(20,30,40),
  Epochs=c(200,300,400)
)
#create training and test set i=1
setwd('C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA')
set.seed(123)
ntest<-floor(nrow(mydata)/5)
trainIndex <- sample(1:nrow(mydata), nrow(mydata)-ntest)
testIndex <- setdiff(1:nrow(mydata), trainIndex)
trainData <- mydata[trainIndex,]
testData <- mydata[testIndex,]
#write.csv(trainData,'trainData.csv')
#write.csv(testData,'testData.csv')
source('C:/Users/mde4023/downloads/StackedSurvivalData/TCGA/CreateData.R')
trainData=read.csv('trainData.csv')[,-1]
testData=read.csv('testData.csv')[,-1]
names(testData)==names(trainData)
# Extract x, time, and status i=1
x <- as.data.frame(trainData[, !(names(trainData) %in% c("time", "status"))])
y <- Surv(trainData$time, trainData$status)

dim(x)           # should be 208 x 1004
length(y)        # should be 208
anyNA(x)         # should be FALSE
anyNA(y)         # should be FALSE
all(sapply(x, is.numeric))  # should be TRUE i=1

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

#
set.seed(777)
DataSeg_t=CreateStack(trainData,nrep =10,myseed=777)
finetrain=DataSeg_t$stacked_df
result=data.frame()
nrow(grid)
for(i in 1:nrow(grid)){
values=grid[i,]
dropout <- as.numeric(values[[1]])
num_nodes <- as.vector(unlist(values[[2]]))
activation <- as.character(values[[3]])
learning_rate <- as.numeric(values[[4]])
patience <- as.integer(values[[5]])
epochs <- as.integer(values[[6]])
seed_R=777
set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
deepsurv_model <- deepsurv(
  data=finetrain,
  time_variable = "time",
  status_variable = "status",
  frac = 0.6,
  activation = activation,
  num_nodes = num_nodes,
  dropout = dropout,
  learning_rate = learning_rate,
  early_stopping = TRUE,
  epochs = epochs,
  patience = patience,
  batch_norm = TRUE,
  batch_size = 128L,
  shuffle = TRUE
)
dnnPred <- predict(deepsurv_model, type="risk",newdata = testData)
dnnC<-UnoC(testData$time,testData$status,dnnPred)
newres <- data.frame(
  dropout = dropout,
  num_nodes = I(list(num_nodes)),  # keep as list to preserve vector
  activation = activation,
  learning_rate = learning_rate,
  patience = patience,
  epochs = epochs,
  cindex = dnnC
)
result=rbind(result,newres)
print(i/nrow(grid))
}
sort(result$cindex,decreasing =T)
select=which(result$cindex==max(result$cindex))
final=result[select,]
final
#dropout    num_nodes activation learning_rate patience epochs   cindex
#246      0.1 32, 64, ....       relu         0.001       30    200 0.731576
final$num_nodes
#[1]   16 32 64


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
