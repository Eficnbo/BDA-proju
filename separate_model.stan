data {
  int<lower=0> N; //number of observations
  int<lower=0> D; //number of dimensions in data excluding chest pain type
  int<lower=0> P; //number of predictions
  int<lower=0,upper=1> y[N]; //targets
  
  int<lower=1,upper=4> cp[N]; //chest pain types (1,2,3,4) (hierarchical groups)
  int<lower=1,upper=4> pred_cp[P]; //chest pain types in prediction data
  
  row_vector[D] x[N]; //data
  row_vector[D] pred_x[P]; //prediction data
}

parameters {
  vector[4] mu_0;
  vector<lower=0>[4] sigma_0;
  real beta_0[4];
  vector[4] mu[D];
  vector<lower=0>[4] sigma[D];
  vector[D] beta[4];
}

model {
  //priors
  for (i in 1:4) {
      mu_0[i] ~ normal(0, 100);
      sigma_0[i] ~ inv_chi_square(0.1);
      beta_0[i] ~ normal(mu_0[i], sigma_0[i]);
  }
  for (d in 1:D) {
      for (i in 1:4) {
          mu[d][i] ~ normal(0, 100);
          sigma[d][i] ~ inv_chi_square(0.1);
          beta[i,d] ~ normal(mu[d][i], sigma[d][i]);
      }
  }

  //likelihood
  for (n in 1:N)
      y[n] ~ bernoulli(inv_logit(beta_0[cp[n]] + x[n] * beta[cp[n]]));
}

generated quantities {
  vector[N] log_lik; //log likelihoods
  int<lower=0,upper=1> y_pred[P]; //predictions
  
  //make predictions on new data
  for (p in 1:P)
      y_pred[p] = bernoulli_rng(inv_logit(beta_0[pred_cp[p]] + x[p] * beta[pred_cp[p]]));
  
  //calculate log likelihood for model evaluation
  for (n in 1:N)
      log_lik[n] = bernoulli_lpmf(y[n] | inv_logit(beta_0[cp[n]] + x[n] * beta[cp[n]]));
}