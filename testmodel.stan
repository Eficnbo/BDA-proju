// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N; // Number of observations
  int<lower=0> M; // Number of features
  int<lower=0> P; // Number of predictions
  int<lower=0,upper=1> y[N]; // Target
  row_vector[M] x[N]; // Variables
  row_vector[M] pred_x[P]; // Prediction variables
}

parameters {
  real alpha;
  vector[M] beta;
  // real mu[M];
  // real<lower=0> sigma[M];
}

model {

  alpha ~ normal(0, 100);
  for (m in 1:M) {
    // mu[m] ~ normal(0, 100);
    // sigma[m] ~ inv_chi_square(0.1);
    beta[m] ~ normal(0, 100);
  }
  
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(alpha + x[n] * beta);
  }
  
}

generated quantities {
  int<lower=0,upper=1> y_pred[P];
  vector[N] log_lik;
  for (p in 1:P) {
    y_pred[p] = bernoulli_rng(inv_logit(alpha + pred_x[p]*beta));
  }
  for (n in 1:N)
    log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha + x[n] * beta);
}

