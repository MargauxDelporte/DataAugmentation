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
source('C:/Users/mde4023/downloads/StackedSurvivalData/TCGA/CreateData.R')
use_condaenv("C:/Users/mde4023/AppData/Local/anaconda3/envs/r-tf", required = TRUE)

# Creating dummy variables
data$Sex_M <- ifelse(data$gender == "male", 1, 0)
data$radioT_Y <- ifelse(data$radiation_therapy == "yes", 1, 0)
data$r1 <- ifelse(data$residual_tumor == "r1", 1, 0)
data$r2 <- ifelse(data$residual_tumor == "r2", 1, 0)
data$rx <- ifelse(data$residual_tumor == "rx", 1, 0)


#########################################
####neural networks#####################
######################################## nrep=1

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
setwd('C:/Users/mde4023/documents/github/StackedSurvivalData/TCGA')
set.seed(123)
ntest<-floor(nrow(mydata)/5)
trainIndex <- sample(1:nrow(mydata), nrow(mydata)-ntest)
testIndex <- setdiff(1:nrow(mydata), trainIndex)
trainData <- mydata[trainIndex,]
testData <- mydata[testIndex,]
#write.csv(trainData,'trainData.csv')
#write.csv(testData,'testData.csv')
trainData=read.csv('trainData.csv')
testData=read.csv('testData.csv')

# Extract x, time, and status i=1
x <- as.data.frame(trainData[, !(names(trainData) %in% c("time", "status"))])
y <- Surv(trainData$time, trainData$status)

dim(x)           # should be 208 x 1004
length(y)        # should be 208
anyNA(x)         # should be FALSE
anyNA(y)         # should be FALSE
all(sapply(x, is.numeric))  # should be TRUE

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
  deepsurv_model <- deepsurv(
    x = x,
    y = y,
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
#dropout    num_nodes activation learning_rate patience epochs    cindex
#1988     0.3 32, 64, ....    sigmoid         0.001       40    400 0.7864761
final$num_nodes
#[1]  32  64 128 256
