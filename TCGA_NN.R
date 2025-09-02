# Load the packages
library(survival)   # For survival analysis functions
library(correctedC)
library(survminer)
library(dplyr)
library(tableone)
library(survivalmodels)
library(intsurv)
library(devtools)
library(keras)
library(fastDummies)
library(purrr)
library(future.apply)
library(reticulate)

py_bin <- "C:/Users/mde4023/AppData/Local/anaconda3/envs/r-tf/python.exe"
stopifnot(file.exists(py_bin))

# Make all workers inherit this Python
Sys.setenv(RETICULATE_PYTHON = py_bin)

# Optional (but nice): initialize in the main session too
use_python(py_bin, required = TRUE)
py_config()
.init_py <- function() {
  # This forces reticulate to bind to the already-pinned interpreter
  reticulate::py_config()
  invisible(TRUE)
}
use_python_env <- function() {
  reticulate::py_config()  # already pinned via RETICULATE_PYTHON
  invisible(TRUE)
}
Sys.setenv(
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

setwd('C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA')
trainData=read.csv('trainData.csv')[,-1]
testData=read.csv('testData.csv')[,-1]
names(trainData)==names(testData)
dim(testData)           # should be 208 x 1004
dim(trainData)           # should be 208 x 1004
length(testData)        # should be 208
anyNA(testData)         # should be FALSE
anyNA(testData)         # should be FALSE
all(sapply(testData, is.numeric))  # should be TRUE



# ---- create the stacking ----
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

# Predictors
length(names(trainData))
all_vars <- names(trainData)[3:1006]
formula_str <- paste("Surv(time, status) ~", paste(all_vars, collapse = " + "))
myformula <- as.formula(formula_str)
head(trainData)

# ---- Parallel plan ----
n_cores <- 20
plan(multisession, workers = n_cores)

# ---- Python env init ----
use_python_env <- function() {
  use_condaenv("C:/Users/mde4023/AppData/Local/anaconda3/envs/r-tf", required = TRUE)
}

# ---- run the neural network ----
##do the replication nrep=10 seedNum=123
simFunction<-function(seedNum,nrep){
  DataSeg_n=CreateStack(trainData,nrep =nrep,myseed=seedNum)
  DataSeg_1=CreateStack(trainData,nrep =1,myseed=seedNum)
  seed_R=seedNum
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel1 <-deepsurv(data=DataSeg_n$trainData,
                       time_variable = "time",
                       status_variable = "status",
                       learning_rate=0.0004999304,
                       lr_decay= 0.0004162785,
                       momentum=0.9417035,
                       #L2_reg=  1.484671e-06,
                       frac=0.7220166,    
                       activation="relu",    
                       num_nodes=c(96,  24), 
                       dropout= 0.07450545,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=30L,
                       batch_norm = T,
                       batch_size=64,
                       shuffle=TRUE)
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel2 <-deepsurv(data=DataSeg_n$replicated_df,
                       time_variable = "time",
                       status_variable = "status",
                       learning_rate=0.0004999304,
                       lr_decay= 0.0004162785,
                       momentum=0.9417035,
                       #L2_reg=  1.484671e-06,
                       frac=0.7220166,    
                       activation="relu",    
                       num_nodes=c(96,  24), 
                       dropout= 0.07450545,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=30L,
                       batch_norm = T,
                       batch_size=64,
                       shuffle=TRUE)
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel3 <-deepsurv(data=DataSeg_n$stacked_df,
                       time_variable = "time",
                       status_variable = "status",
                       learning_rate=0.0004999304,
                       lr_decay= 0.0004162785,
                       momentum=0.9417035,
                       #L2_reg=  1.484671e-06,
                       frac=0.7220166,    
                       activation="relu",    
                       num_nodes=c(96,  24), 
                       dropout= 0.07450545,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=30L,
                       batch_norm = T,
                       batch_size=64,
                       shuffle=TRUE)
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel4 <-deepsurv(data=DataSeg_1$replicated_df,
                       time_variable = "time",
                       status_variable = "status",
                       learning_rate=0.0004999304,
                       lr_decay= 0.0004162785,
                       momentum=0.9417035,
                       #L2_reg=  1.484671e-06,
                       frac=0.7220166,    
                       activation="relu",    
                       num_nodes=c(96,  24), 
                       dropout= 0.07450545,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=30L,
                       batch_norm = T,
                       batch_size=64,
                       shuffle=TRUE)
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  dnnModel5 <-deepsurv(data=DataSeg_1$stacked_df,
                       time_variable = "time",
                       status_variable = "status",
                       learning_rate=0.0004999304,
                       lr_decay= 0.0004162785,
                       momentum=0.9417035,
                       #L2_reg=  1.484671e-06,
                       frac=0.7220166,    
                       activation="relu",    
                       num_nodes=c(96,  24), 
                       dropout= 0.07450545,          
                       early_stopping=TRUE, 
                       epochs=1000L,
                       patience=30L,
                       batch_norm = T,
                       batch_size=64,
                       shuffle=TRUE)
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
  r=c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5)
#for cox: use cox lasso in seperate script
  write.csv(t(r),file=paste0('C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA/resultNN/NN_',seedNum,'.csv'))
  
  return(c(seedNum,dnnC1,dnnC2,dnnC3,dnnC4,dnnC5))
}

library(future)
library(future.apply)

# Reasonable core count
# ncores <- max(1, parallel::detectCores() - 1)
plan(multisession, workers = 25)

safe_sim <- function(seedNum, nrep) {
  tryCatch({
    .init_py()   # <- key line
    # ensure outdir exists (prevents silent write.csv failures)
    out_dir <- "C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA/resultNN"
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    
    as.numeric(simFunction(seedNum = seedNum, nrep = nrep))
  }, error = function(e) {
    warning(sprintf("seed %s failed: %s", seedNum, e$message))
    c(as.numeric(seedNum), rep(NA_real_, 5))  # always fixed length
  })
}

results_list <- future_lapply(
  seeds,
  function(s) safe_sim(s, nrep),
  future.seed = TRUE
)

results_matrix <- do.call(rbind, results_list)
colnames(results_matrix) <- c("seed","dnnC1","dnnC2","dnnC3","dnnC4","dnnC5")

write.csv(
  results_matrix,
  "C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA/resultNN/NN_all.csv",
  row.names = FALSE
)

results_matrix=as.data.frame(results_matrix)
results_matrix$dnnC1
sum(is.na(results_matrix$dnnC1))


seeds <- 1001:1038
nrep  <- 5

plan(multisession, workers = 38/2)

safe_sim <- function(seedNum, nrep) {
  tryCatch({
    .init_py()   # <- key line
    # ensure outdir exists (prevents silent write.csv failures)
    out_dir <- "C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA/resultNN"
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    
    as.numeric(simFunction(seedNum = seedNum, nrep = nrep))
  }, error = function(e) {
    warning(sprintf("seed %s failed: %s", seedNum, e$message))
    c(as.numeric(seedNum), rep(NA_real_, 5))  # always fixed length
  })
}

results_list <- future_lapply(
  seeds,
  function(s) safe_sim(s, nrep),
  future.seed = TRUE
)

##check if everything is fitted and calculate result
# Directory path
path <- "C:/Users/mde4023/Downloads/StackedSurvivalData/TCGA/resultNN/"

files <- list.files(path, full.names = TRUE)
length(files)
all_data <- do.call(rbind, lapply(files, read.csv))
colnames(all_data) <- c("seed","dnnC1","dnnC2","dnnC3","dnnC4","dnnC5")
sum(is.na(all_data$dnnC1))
all_data=subset(all_data,!is.na(all_data$dnnC1))
# If you want to keep track of which file each row came from
all_data <- lapply(files, function(f) {
  df <- read.csv(f)
  df$file_source <- basename(f)
  df
}) %>% bind_rows()
head(all_data)
round(colMeans(all_data[1:999,3:7]),3)














































