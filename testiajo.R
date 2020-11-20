library(rstan)
library(tidyverse)

data <- read.csv('heart.csv', sep=',')
s <- sample(nrow(data))
shuffled <- data[s,]

data_train <- head(shuffled, 270)
data_test <- tail(shuffled, 33)
x_train <- select(data_train,'age':'thal')
x_test <- select(data_test,'age':'thal')

standata <- list(y=data_train$target, N=nrow(data_train), M=ncol(data_train)-1, P=nrow(data_test), x=x_train, pred_x=x_test)

a <- stan(file = 'testmodel.stan', data = standata)
b <- monitor(a)
c <- rstan::extract(a)
plot(c$y_pred)
hist(c$y_pred)
hist(data$target)
d <- list()
drwas <- as.data.frame(a)
calc <- 0
drwas
abc <- select(drwas,'y_pred[1]':'y_pred[33]')
abc <- as.matrix(abc,ncols=33)
ii <- 1
for(i in 1:33) {
  
  d[ii] <- mean(abc[,i])
  ii= ii+1
}

totta = list()
uusi <- data_test$target

for(k in 1:33) {
  if(  (d[k]> 0.5 && uusi[k]==1) || ( d[k] < 0.5 && uusi[k] ==0) ) {
    totta[k] <- TRUE
  }
  
}


