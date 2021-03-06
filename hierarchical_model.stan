data {
  int<lower=0> N; //number of observations
  int<lower=0> D; //number of dimensions in data excluding chest pain type
  int<lower=0> P; //number of predictions
  int<lower=0,upper=1> y[N]; //targets
  int<lower=1,upper=4> cp[N]; //chest pain types (1,2,3,4) (hierarchical groups)
  int<lower=1,upper=4> pred_cp[P]; //chest pain types in prediction data
  vector[D] x[N]; //data
  vector[D] pred_x[P]; //prediction data
}

parameters {
  real mu_0;
  real<lower=0> sigma_0;
  real beta_0[4];
  real mu[D];
  real<lower=0> sigma[D];
  row_vector[D] beta[4];
}

model {
  //priors
  mu_0 ~ normal(0, 100);
  sigma_0 ~ inv_chi_square(0.1);
  for (i in 1:4)
      beta_0[i] ~ normal(mu_0, sigma_0);
      
  mu[1] ~ normal(0, 100);
  mu[2] ~ normal(0, 100);
  mu[3] ~ normal(0, 0.01);
  mu[4] ~ normal(0, 0.01);
  mu[5] ~ normal(0, 0.01);
  mu[6] ~ normal(0, 0.01);
  mu[7] ~ normal(0, 0.01);
  mu[8] ~ normal(0, 100);
  sigma[1] ~ inv_chi_square(0.1);
  sigma[2] ~ inv_chi_square(0.1);
  sigma[3] ~ inv_chi_square(100);
  sigma[4] ~ inv_chi_square(100);
  sigma[5] ~ inv_chi_square(100);
  sigma[6] ~ inv_chi_square(100);
  sigma[7] ~ inv_chi_square(100);
  sigma[8] ~ inv_chi_square(0.1);

  for (d in 1:D) {
      for (i in 1:4)
          beta[i,d] ~ normal(mu[d], sigma[d]);
  }

  //likelihood
  for (n in 1:N)
      y[n] ~ bernoulli_logit(beta_0[cp[n]] + beta[cp[n]] * x[n]);
}

generated quantities {
  vector[N] log_lik; //log likelihoods
  int<lower=0,upper=1> y_pred[P]; //predictions
  
  //predict target in test set
  for (p in 1:P)
      y_pred[p] = bernoulli_logit_rng(beta_0[pred_cp[p]] + beta[pred_cp[p]] * pred_x[p]);
  
  //calculate log likelihoods for model evaluation
  for (n in 1:N)
      log_lik[n] = bernoulli_logit_lpmf(y[n] | beta_0[cp[n]] + beta[cp[n]] * x[n]);
}
