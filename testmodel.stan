// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0,upper=1> y[N]; // target
  row_vector[M] x[N]; // variables
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real alpha;
  vector[M] beta; //slope
  real mu[M];
  real<lower=0> sigma[M];
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  alpha ~ normal(0, 100);
  for (m in 1:M) {
    mu[m] ~ normal(0, 100);
    sigma[m] ~ inv_chi_square(0.1);
  }
  
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(alpha + x[n] * beta);
  }
  
}

