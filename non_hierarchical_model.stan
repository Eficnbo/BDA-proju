data {
  int<lower=0> N; //number of observations
  int<lower=0> D; //number of dimensions in data excluding chest pain type
  int<lower=0> P; //number of predictions
  int<lower=0,upper=1> y[N]; //targets
  int<lower=1,upper=4> cp[N]; //chest pain types (1,2,3,4)
  int<lower=1,upper=4> pred_cp[P]; //chest pain types in prediction data
  vector[D] x[N]; //data
  vector[D] pred_x[P]; //prediction data
}

parameters {
  real mu_0;
  real<lower=0> sigma_0;
  real beta_0[4];
  real mu;
  real<lower=0> sigma;
  row_vector[D] beta;
}

model {
  //priors
  for (i in 1:4)
      beta_0[i] ~ normal(0, 100);
      
  for (d in 1:D)
      beta[d] ~ normal(0, 100);

  //likelihood
  for (n in 1:N)
      y[n] ~ bernoulli_logit(beta_0[cp[n]] + beta * x[n]);
}

generated quantities {
  vector[N] log_lik; //log likelihoods
  int<lower=0,upper=1> y_pred[P]; //predictions
  
  //predict target in test set
  for (p in 1:P)
      y_pred[p] = bernoulli_logit_rng(beta_0[pred_cp[p]] + beta * pred_x[p]);
  
  //calculate log likelihoods for model evaluation
  for (n in 1:N)
      log_lik[n] = bernoulli_logit_lpmf(y[n] | beta_0[cp[n]] + beta * x[n]);
}
