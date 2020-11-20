data {
  int<lower=0> N; //number of observations
  int<lower=0> J; //nuber of features
  vector[N] x[J];
  int<lower=0,upper=1> y[N];
}
parameters {
  real alpha;
  vector[J] beta;
}
model {
   for(j in 1:J) {
      beta[J] ~ normal(0, 100)
   }
   y ~ bernoulli_logit(alpha + beta[1]*x[1] + beta[2]*x[2] + beta[3]*x[3]);
}