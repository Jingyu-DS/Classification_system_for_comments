library(data.table)
library(tidyverse)
library(Rtsne)
library(ggplot2)
library(caret)
library(ggplot2)
library(ClusterR)
library(dplyr)
library(Metrics)
library(xgboost)

setwd("~/Desktop/final project 380")

train_tsne <- fread("./project/volume/data/interim/train_tsne.csv")
test_tsne <- fread("./project/volume/data/interim/test_tsne.csv")

y.train<-train_tsne$Reddit
dtrain_pre <- as.matrix(train_tsne[,c(3,4)])
dtrain <- xgb.DMatrix(dtrain_pre,label=y.train,missing=NA)
dtest <- xgb.DMatrix(as.matrix(test_tsne[,c(3,4)]),missing=NA)


#### Xgboost ####
hyper_perm_tune<-NULL
num_class <- length(levels(as.factor(y.train)))
param <- list(  gamma               = 0.02,  
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                eta                 = 0.01,   #0.02
                max_depth           = 8,   
                min_child_weight    = 1,
                subsample           = 0.75,  
                colsample_bytree    = 1.0,  
                objective="multi:softprob",
                tree_method = 'hist',
                num_class=num_class
)


XGBm<-xgb.cv( params=param,nfold=25,nrounds=10000,missing=NA,data=dtrain,print_every_n=1,early_stopping_rounds=25)

best_ntrees<-unclass(XGBm)$best_iteration

new_row<-data.table(t(param))

new_row$best_ntrees<-best_ntrees

test_error<-unclass(XGBm)$evaluation_log[best_ntrees,]$test_mlogloss_mean
new_row$test_error<-test_error
# hyper_perm_tune<-rbind(new_row,hyper_perm_tune)

watchlist <- list( train = dtrain)

XGBm<-xgb.train( params=param,nrounds=best_ntrees,missing=NA,data=dtrain,watchlist=watchlist,print_every_n=1)


xgb.pred = predict(XGBm,dtest)
# xgb.test = predict(XGBm,dtrain)
# test_matrix <- matrix(xgb.test, ncol=10, byrow=T)
# test_table <- data.table(test_matrix)
pred_matrix <- matrix(xgb.pred, ncol=10, byrow=T)
pred <- data.table(pred_matrix)

sub_sam <- fread("./project/volume/data/raw/example_sub.csv")
sub_sam[,2:11] <- pred[,1:10]

fwrite(sub_sam, "./project/volume/data/processed/submission14.csv")
