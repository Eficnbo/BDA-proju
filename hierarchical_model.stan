data {
  int<lower=0> N; //Number of observations
  int<lower=0> P; // Number of predictions
  int<lower=0,upper=1> y[N]; //targets
  
  //cps are 1,2,3 or 4 in this model but 0,1,2 or 3 in data. => Add 1 to each cp
  int<lower=1,upper=4> cp[N]; //chest pain types
  int<lower=1,upper=4> cp_pred[P]; //chest pain types
  
  real x1[N]; //feature 1
  real x2[N]; //feature 2
  real x3[N]; //feature 3
  real x4[N]; //feature 4
  real x5[N]; //feature 5
  real x6[N]; //feature 6
  real x7[N]; //feature 7
  real x8[N]; //feature 8
  
  real x1_pred[P]; //feature 1
  real x2_pred[P]; //feature 2
  real x3_pred[P]; //feature 3
  real x4_pred[P]; //feature 4
  real x5_pred[P]; //feature 5
  real x6_pred[P]; //feature 6
  real x7_pred[P]; //feature 7
  real x8_pred[P]; //feature 8
}

parameters {
  real mu_0;
  real mu_1;
  real mu_2;
  real mu_3;
  real mu_4;
  real mu_5;
  real mu_6;
  real mu_7;
  real mu_8;
  real<lower=0> sigma_0;
  real<lower=0> sigma_1;
  real<lower=0> sigma_2;
  real<lower=0> sigma_3;
  real<lower=0> sigma_4;
  real<lower=0> sigma_5;
  real<lower=0> sigma_6;
  real<lower=0> sigma_7;
  real<lower=0> sigma_8;
  vector[4] beta_0;
  vector[4] beta_1;
  vector[4] beta_2;
  vector[4] beta_3;
  vector[4] beta_4;
  vector[4] beta_5;
  vector[4] beta_6;
  vector[4] beta_7;
  vector[4] beta_8;
}

model {
  //priors
  mu_0 ~ normal(0, 100);
  mu_1 ~ normal(0, 100);
  mu_2 ~ normal(0, 100);
  mu_3 ~ normal(0, 100);
  mu_4 ~ normal(0, 100);
  mu_5 ~ normal(0, 100);
  mu_6 ~ normal(0, 100);
  mu_7 ~ normal(0, 100);
  mu_8 ~ normal(0, 100);
  sigma_0 ~ inv_chi_square(0.1);
  sigma_1 ~ inv_chi_square(0.1);
  sigma_2 ~ inv_chi_square(0.1);
  sigma_3 ~ inv_chi_square(0.1);
  sigma_4 ~ inv_chi_square(0.1);
  sigma_5 ~ inv_chi_square(0.1);
  sigma_6 ~ inv_chi_square(0.1);
  sigma_7 ~ inv_chi_square(0.1);
  sigma_8 ~ inv_chi_square(0.1);
  for (i in 1:4) {
      beta_0[i] ~ normal(mu_0, sigma_0);
      beta_1[i] ~ normal(mu_1, sigma_1);
      beta_2[i] ~ normal(mu_2, sigma_2);
      beta_3[i] ~ normal(mu_3, sigma_3);
      beta_4[i] ~ normal(mu_4, sigma_4);
      beta_5[i] ~ normal(mu_5, sigma_5);
      beta_6[i] ~ normal(mu_6, sigma_6);
      beta_7[i] ~ normal(mu_7, sigma_7);
      beta_8[i] ~ normal(mu_8, sigma_8);
  }

  //likelihood
  for (n in 1:N)
    y[n] ~ bernoulli(inv_logit(beta_0[cp[n]] + beta_1[cp[n]] * x1[n] + beta_2[cp[n]] * x2[n] + beta_3[cp[n]] * x3[n] + beta_4[cp[n]] * x4[n] + beta_5[cp[n]] * x5[n] + beta_6[cp[n]] * x6[n] + beta_7[cp[n]] * x7[n] + beta_8[cp[n]] * x8[n]));
}

generated quantities {
  int<lower=0,upper=1> y_pred[P];
  vector[N] log_lik;
  for (p in 1:P) {
    y_pred[p] = bernoulli_rng(inv_logit(beta_0[cp_pred[p]] + beta_1[cp_pred[p]] * x1_pred[p] + beta_2[cp_pred[p]] * x2_pred[p] + beta_3[cp_pred[p]] * x3_pred[p] + beta_4[cp_pred[p]] * x4_pred[p] + beta_5[cp_pred[p]] * x5_pred[p] + beta_6[cp_pred[p]] * x6_pred[p] + beta_7[cp_pred[p]] * x7_pred[p] + beta_8[cp_pred[p]] * x8_pred[p]));
  }
  
  for (n in 1:N)
    log_lik[n] = bernoulli_lpmf(y[n] | inv_logit(beta_0[cp[n]] + beta_1[cp[n]] * x1[n] + beta_2[cp[n]] * x2[n] + beta_3[cp[n]] * x3[n] + beta_4[cp[n]] * x4[n] + beta_5[cp[n]] * x5[n] + beta_6[cp[n]] * x6[n] + beta_7[cp[n]] * x7[n] + beta_8[cp[n]] * x8[n]));


}

