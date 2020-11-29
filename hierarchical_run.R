library(rstan)
library(loo)
library(tidyverse)

data <- read.csv('heart.csv', sep=',')

N <- nrow(data)

good_data <- data.frame(matrix(0L, ncol = 10, nrow = N))
columns <- c("age", "sex", "cp", "trestbps", "chol", "fbs",
             "thalach", "exang", "oldpeak", "target")

colnames(good_data) <- columns
good_data$age <- data$ï..age
good_data$sex <- data$sex
good_data$cp <- data$cp + 1
good_data$trestbps <- data$trestbps
good_data$chol <- data$chol
good_data$fbs <- data$fbs
good_data$thalach <- data$thalach
good_data$exang <- data$exang
good_data$oldpeak <- data$oldpeak
good_data$target <- data$target


# PREDICTING

s <- sample(nrow(good_data))
shuffled <- good_data[s,]

N_test <- 33

data_train <- head(shuffled, N - N_test)
data_test <- tail(shuffled, N_test)
x_train <- select(data_train,'age','sex','trestbps':'oldpeak')
x_test <- select(data_test,'age','sex','trestbps':'oldpeak')
#x_train <- select(data_train,'age')
#x_test <- select(data_test,'age')

#standata <- list(y=data_train$target, N=nrow(data_train), P=nrow(data_test), x1=x_train$age, x2=x_train$sex, x3=x_train$trestbps, x4=x_train$chol, x5=x_train$fbs, x6=x_train$thalach, x7=x_train$exang, x8=x_train$oldpeak,
#                 x1_pred=x_test$age, x2_pred=x_test$sex, x3_pred=x_test$trestbps, x4_pred=x_test$chol, x5_pred=x_test$fbs, x6_pred=x_test$thalach, x7_pred=x_test$exang, x8_pred=x_test$oldpeak,
#                 cp=data_train$cp, cp_pred=data_test$cp)

standata <- list(y=data_train$target, D=ncol(x_train), N=nrow(data_train), P=nrow(data_test), cp=data_train$cp, pred_cp=data_test$cp, x=x_train, pred_x=x_test)

model <- stan(file = 'hierachical_model2.stan', data = standata)
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
