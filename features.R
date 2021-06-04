library(data.table)

setwd("~/Desktop/final project 380")

train_group <- fread("./project/volume/data/raw/training_data.csv")

# Indicate the Reddit 
train_group$Reddit <- names(train_group[,-c(1,2)])[max.col(train_group[,-c(1,2)])]
train_group$Reddit<-recode(train_group$Reddit, 'subredditcars'=0, 'subredditCooking'=1, 
                           'subredditMachineLearning'=2, 'subredditmagicTCG'=3, "subredditpolitics"=4,
                           "subredditReal_Estate"=5, "subredditscience"=6, "subredditStockMarket"=7, 
                           "subreddittravel"=8, "subredditvideogames"=9)

train_id <- train_group[,c(1,13)]

train_emb <- fread("./project/volume/data/raw/training_emb.csv")
train <- cbind(train_id, train_emb)

test_id <- fread("./project/volume/data/raw/test_file.csv")
test_emb <- fread("./project/volume/data/raw/test_emb.csv")

test_id$Reddit <- NA
test <- cbind(test_id[,c(1,3)], test_emb)

universe <- rbind(train, test)

#### PCA ####
pca_data <- universe[,-c(1,2)]
# do a pca
pca<-prcomp(pca_data)
screeplot(pca)
pca_dt<-data.table(unclass(pca)$x)

#### Tsne ####
tsne<-Rtsne(pca_dt,pca = F,perplexity=100,check_duplicates = F)
tsne_dt<-data.table(tsne$Y)

universe_tsne <- cbind(universe[,c(1,2)], tsne_dt)
train_tsne <- universe_tsne[1:200]
test_tsne <- universe_tsne[201:20755]

fwrite(train_tsne, "./project/volume/data/interim/train_tsne.csv")
fwrite(test_tsne, "./project/volume/data/interim/test_tsne.csv")
