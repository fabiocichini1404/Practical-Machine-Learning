# Machine Learning project
# Loading libraries
library(caret)
library(corrplot)
library(dplyr)
library(ggplot2)
library(gbm)
library(randomForest)
library(rattle)
library(RColorBrewer)
library(rpart)
library(rpart.plot)


#Loading data
train_1 <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),header=TRUE)
valid_1 <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"),header=TRUE)
dim(train_1)
dim(valid_1)


#Cleaning data (Removing NAs and 7 first collumns because they have low importante in outcome)
train_data<- train_1[, colSums(is.na(train_1)) == 0]
train_data <- train_data[, -c(1:7)]
valid_data <- valid_1[, colSums(is.na(valid_1)) == 0]
valid_data <- valid_data[, -c(1:7)]
dim(train_data)
dim(valid_data)


#Creating data partition
set.seed(12345) 
Partition <- createDataPartition(train_data$classe, p = 0.7, list = FALSE)
train_data <- train_data[Partition, ]
test_data <- train_data[-Partition, ]
dim(train_data)
dim(test_data)


# Removing variables that are NZV (Near Zero Variance)
NearZeroVariance <- nearZeroVar(train_data)
train_data <- train_data[, -NearZeroVariance]
test_data  <- test_data[, -NearZeroVariance]
dim(train_data)
dim(test_data)


# Modeling the data using classification trees
set.seed(12345)
Treemodel <- rpart(classe ~ ., data=train_data, method="class")
fancyRpartPlot(Treemodel)


# Testing with classification trees
Treemodeltest <- predict(Treemodel, test_data, type = "class")
Treematrixresults <- confusionMatrix(Treemodeltest, test_data$classe)
Treematrixresults


# Modeling the data using Random forest
set.seed(12345)
Ranforestmodel <- trainControl(method="cv", number=3, verboseIter=FALSE)
RanforestmodelX <- train(classe ~ ., data=train_data, method="rf", trControl=Ranforestmodel)
RanforestmodelX$finalModel


# Testing with Random forest
Ranforesttest <- predict(RanforestmodelX, newdata=test_data)
Ranforestresults <- confusionMatrix(Ranforesttest, test_data$classe)
Ranforestresults


# Applying  models to valid_data

# BEST RESULT!! (got with Random forest Model)
FinaltestRF <- predict(RanforestmodelX, newdata=valid_data)
FinaltestRF


#2° result (got with Classification trees model)
FinaltestTree <- predict(Treemodel, newdata=valid_data)
FinaltestTree<-apply(FinaltestTree,1,which.max)
FinaltestTree