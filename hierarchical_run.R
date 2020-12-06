library(rstan)
library(loo)
library(tidyverse)
library(ggcorrplot)

data <- read.csv('heart.csv', sep=',')
N <- nrow(data)

columns <- c("age", "sex", "cp", "trestbps", "chol", "fbs",
             "thalach", "exang", "oldpeak", "target")

data_std <- data.frame(matrix(0L, ncol = 10, nrow = N))
colnames(data_std) <- columns
data_std$age <- (data$ï..age - mean(data$ï..age))/sd(data$ï..age)
data_std$sex <- data$sex
data_std$cp <- data$cp + 1
data_std$trestbps <- (data$trestbps - mean(data$trestbps))/sd(data$trestbps)
data_std$chol <- (data$chol - mean(data$chol))/sd(data$chol)
data_std$fbs <- data$fbs
data_std$thalach <- (data$thalach - mean(data$thalach))/sd(data$thalach)
data_std$exang <- data$exang
data_std$oldpeak <- (data$oldpeak - mean(data$oldpeak))/sd(data$oldpeak)
data_std$target <- data$target


ggcorrplot(cor(data_std))

# PREDICTING

s <- sample(nrow(data_std))
shuffled <- data_std[s,]

N_test <- 33

data_train <- head(shuffled, N - N_test)
data_test <- tail(shuffled, N_test)
x_train <- select(data_train,'age','sex','trestbps':'oldpeak')
x_test <- select(data_test,'age','sex','trestbps':'oldpeak')

standata <- list(y=data_train$target, D=ncol(x_train), N=nrow(data_train),
                 P=nrow(data_test), cp=data_train$cp, pred_cp=data_test$cp,
                 x=x_train, pred_x=x_test)

hierarchical_model <- stan(file = 'hierarchical_model.stan', data = standata,
                           iter = 2000, control = list(adapt_delta = 0.999))
hie_mon <- monitor(hierarchical_model)

hierarchical_predictions <- data.frame(matrix(0L, ncol = 3, nrow = N_test))
cols <- c("mu", "prediction", "target")
colnames(hierarchical_predictions) <- cols
hierarchical_predictions$target <- data_test$target

hierarchical_draws <- as.data.frame(hierarchical_model)
hierarchical_preds <- as.matrix(select(hierarchical_draws,'y_pred[1]':'y_pred[33]'), ncols=N_test)

hie_wrong <- 0
for (i in 1:N_test) {
  mu <- mean(hierarchical_preds[,i])
  hierarchical_predictions$mu[i] <- mu
  if (mu >= 0.5) hierarchical_predictions$prediction[i] <- 1
  if (hierarchical_predictions$prediction[i] != hierarchical_predictions$target[i]) hie_wrong <- hie_wrong + 1
}
hie_wrong

# LOO

hierarchical_loo <- loo(hierarchical_model)
hierarchical_loo$elpd_loo

barplot(hierarchical_loo$diagnostics$pareto_k, ylim=c(-0.2, 0.8))
abline(a=0.7, b=0, col='red')
title('k values')

hierarchical_loo$p_loo


# NON-HIERARCHICAL #############################################################


separate_model <- stan(file = 'non_hierarchical_model.stan', data = standata,
                       iter = 2000, control = list(adapt_delta = 0.999))
sep_mon <- monitor(separate_model)

separate_predictions <- data.frame(matrix(0L, ncol = 3, nrow = N_test))
colnames(separate_predictions) <- cols
separate_predictions$target <- data_test$target

separate_draws <- as.data.frame(separate_model)
separate_preds <- as.matrix(select(separate_draws,'y_pred[1]':'y_pred[33]'), ncols=N_test)

sep_wrong <- 0
for (i in 1:N_test) {
  mu <- mean(separate_preds[,i])
  separate_predictions$mu[i] <- mu
  if (mu >= 0.5) separate_predictions$prediction[i] <- 1
  if (separate_predictions$prediction[i] != separate_predictions$target[i]) sep_wrong <- sep_wrong + 1
}
sep_wrong

# LOO

separate_loo <- loo(separate_model)
separate_loo$elpd_loo

barplot(separate_loo$diagnostics$pareto_k, ylim=c(-0.2, 0.8))
abline(a=0.7, b=0, col='red')
title('k values')

separate_loo$p_loo
