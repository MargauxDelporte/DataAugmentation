library(correctedC)
library(survminer)
library(dplyr)
library(survivalmodels)
library(survival)
library(devtools)
library(randomForestSRC)
library(readr)
#source('C:/Users/mde4023/downloads/StackedSurvivalData/TCGA/CreateData.R')
setwd('C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA')
trainData=read.csv('trainData.csv')
testData=read.csv('testData.csv')

trainData_clean <- trainData[, !names(trainData) %in% "X"]
rfModel1 <- rfsrc(Surv(time,status)~.,data=trainData_clean)

# Define a tuning grid
tune_grid <- expand.grid(
  ntree = c(250, 500),
  mtry = c(10, 30, 50, 100, 200),  # for p = 1000
  nodesize = c(5, 10, 20)
)

# Container for results
results <- data.frame()

# Loop over grid i=1
for (i in seq_len(nrow(tune_grid))) {
  cat("Fitting model", i, "of", nrow(tune_grid), "\n")
  set.seed(123)
  params <- tune_grid[i, ]
  
  model <- rfsrc(
    formula = Surv(time, status) ~ .,
    data = trainData[, !names(trainData) %in% "X"],  # remove character ID
    ntree = params$ntree,
    mtry = params$mtry,
    nodesize = params$nodesize,
    block.size = 1,  # prevents full data in memory
    importance = "none",
    forest = TRUE
  )
  rfPred1 <- predict(model,newdata = testData)
  rfC1<-UnoC(testData$time, testData$status,  rfPred1$predicted)
  
  results <- rbind(results, data.frame(
    ntree = params$ntree,
    mtry = params$mtry,
    nodesize = params$nodesize,
    cindex = rfC1  # last oob error (1 - C-index)
  ))
}
results[which(results$cindex==max(results$cindex)),]

