library(rstan)
library(tidyverse)

data <- read.csv('heart.csv', sep=',')
s <- sample(nrow(data))
shuffled <- data[s,]

data_train <- head(shuffled, 270)
data_test <- tail(shuffled, 33)
x_train <- select(data_train,'ï..age':'thal')
x_test <- select(data_test,'ï..age':'thal')

standata <- list(y=data_train$target, N=nrow(data_train), M=ncol(data)-1, P=nrow(data_test), x=x_train, pred_x=x_test)

a <- stan(file = 'testmodel.stan', data = standata)
b <- monitor(a)
