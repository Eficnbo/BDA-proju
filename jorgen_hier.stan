data {
  int<lower=0> N; //Number of observations
  int<lower=0,upper=1> y[N]; //targets
  
  //cps are 1,2,3 or 4 in this model but 0,1,2 or 3 in data. => Add 1 to each cp
  int<lower=1,upper=4> cp[N]; //chest pain types
  
  real x1[N]; //feature 1
}

parameters {
  real mu_0;
  real mu_1;
  real sigma_0;
  real sigma_1;
  vector[4] beta_0;
  vector[4] beta_1;
}

model {
  //priors
  mu_0 ~ normal(0, 100);
  mu_1 ~ normal(0, 100);
  sigma_0 ~ inv_chi_square(0.1);
  sigma_1 ~ inv_chi_square(0.1);
  for (i in 1:4) {
      beta_0[i] ~ normal(mu_0, sigma_0);
      beta_1[i] ~ normal(mu_1, sigma_1);
  }

  //likelihood
  for (n in 1:N)
    y[n] ~ bernoulli(inv_logit(beta_0[cp[n]] + beta_1[cp[n]] * x1[n]));
}

generated quantities {
  int<lower=0,upper=1> y_pred[N];
  vector[N] log_lik;
  for (p in 1:N) {
    y_pred[p] = bernoulli_rng(inv_logit(beta_0[cp[p]] + beta_1[cp[p]] * x1[p]));
  }
  for (n in 1:N)
      log_lik[n] = bernoulli_lpmf(y[n] | inv_logit(beta_0[cp[n]] + beta_1[cp[n]] * x1[n]));
}