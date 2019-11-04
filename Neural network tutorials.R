## Try my first Neural Network
## Tutorial https://rpubs.com/sediaz/Neural_Network

library(tidyverse)
library(caret)
library(nnet)
library(e1071)

data("iris")
head(iris)

ggplot(data = iris, aes(x = Petal.Width, y = Petal.Length, color = Species)) + 
  geom_point()


## Normalize the data. WERY important in NN
iris_norm <- iris
iris_norm$Species <- as.numeric(iris$Species)
iris_norm <- as.data.frame(apply(iris_norm, 2, function(x) (x - min(x))/(max(x)-min(x))))


##Create trainset and testset
index <- createDataPartition(iris_norm$Species, p = 0.7, list = FALSE)
trainset <- iris_norm[index,]
testset <- iris_norm[-index,]

## set.seed() ensure that random objects  can be reproduced.
set.seed(5)
fit <- nnet(Species ~., data = trainset, linout = TRUE, size = 20)

prediction <- factor(round(fit$fitted.values*2 +1))
real_value <- factor(trainset$Species*2 +1)

postResample(prediction, real_value)

prediction_test <- predict(fit, newdata = testset)
prediction_test <- factor(round(prediction_test*2 +1))
real_value_test <- factor(testset$Species*2 +1)

postResample(prediction_test, real_value_test)

# Now we can compare this result with a knn model
set.seed(5)
my_knn_model <- train(Species ~ Petal.Width + Petal.Length, data = trainset, method = "knn", tunelength = 5)
my_knn_model


prediction_knn <- predict(my_knn_model, newdata = testset)
prediction_knn <- factor(round(prediction_knn*2 +1))
postResample(prediction_knn, real_value_test) ## same result ass KNN

## https://towardsdatascience.com/build-your-own-neural-network-classifier-in-r-b7f1f183261d
## For more complete understanding and built the algo bottom up.
