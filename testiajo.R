library(rstan)
library(loo)
library(tidyverse)

data <- read.csv('heart.csv', sep=',')

N <- nrow(data)

good_data <- data.frame(matrix(0L, ncol = 28, nrow = N))
columns <- c("age", "sex", "cp0", "cp1", "cp2", "cp3", "trestbps", "chol", "fbs",
            "restecg0", "restecg1", "restecg2", "thalach", "exang", "oldpeak", "slope0", "slope1", "slope2",
            "ca0", "ca1", "ca2", "ca3", "ca4", "thal0", "thal1", "thal2", "thal3", "target")

colnames(good_data) <- columns
good_data$age <- data$age
good_data$sex <- data$sex
good_data$trestbps <- data$trestbps
good_data$chol <- data$chol
good_data$fbs <- data$fbs
good_data$thalach <- data$thalach
good_data$exang <- data$exang
good_data$oldpeak <- data$oldpeak
good_data$target <- data$target

for (i in 1:N) {
  if (data$cp[i] == 0) good_data$cp0[i] <- 1
  else if (data$cp[i] == 1) good_data$cp1[i] <- 1
  else if (data$cp[i] == 2) good_data$cp2[i] <- 1
  else good_data$cp3[i] <- 1
  
  if (data$restecg[i] == 0) good_data$restecg0[i] <- 1
  else if (data$restecg[i] == 1) good_data$restecg1[i] <- 1
  else good_data$restecg2[i] <- 1
  
  if (data$slope[i] == 0) good_data$slope0[i] <- 1
  else if (data$slope[i] == 1) good_data$slope1[i] <- 1
  else good_data$slope2[i] <- 1
  
  if (data$ca[i] == 0) good_data$ca0[i] <- 1
  else if (data$ca[i] == 1) good_data$ca1[i] <- 1
  else if (data$ca[i] == 2) good_data$ca2[i] <- 1
  else if (data$ca[i] == 3) good_data$ca3[i] <- 1
  else good_data$ca4[i] <- 1
  
  if (data$thal[i] == 0) good_data$thal0[i] <- 1
  else if (data$thal[i] == 1) good_data$thal1[i] <- 1
  else if (data$thal[i] == 2) good_data$thal2[i] <- 1
  else good_data$thal3[i] <- 1
}


# PREDICTING

s <- sample(nrow(good_data))
shuffled <- good_data[s,]

N_test <- 33

data_train <- head(shuffled, N - N_test)
data_test <- tail(shuffled, N_test)
x_train <- select(data_train,'age':'thal3')
x_test <- select(data_test,'age':'thal3')

standata <- list(y=data_train$target, N=nrow(data_train), M=ncol(data_train)-1, P=nrow(data_test), x=x_train, pred_x=x_test)

model <- stan(file = 'testmodel.stan', data = standata)
mon <- monitor(model)

predictions <- data.frame(matrix(0L, ncol = 3, nrow = N_test))
cols <- c("mu", "prediction", "target")
colnames(predictions) <- cols
predictions$target <- data_test$target

draws <- as.data.frame(model)
preds <- as.matrix(select(draws,'y_pred[1]':'y_pred[33]'), ncols=N_test)

wrong <- 0
for (i in 1:N_test) {
  mu <- mean(preds[,i])
  predictions$mu[i] <- mu
  if (mu >= 0.5) predictions$prediction[i] <- 1
  if (predictions$prediction[i] != predictions$target[i]) wrong <- wrong + 1
}
wrong

# LOO

model_loo <- loo(model)
model_loo$elpd_loo

barplot(model_loo$diagnostics$pareto_k, ylim=c(-0.2, 0.8))
abline(a=0.7, b=0, col='red')
title('k values')

model_loo$p_loo
