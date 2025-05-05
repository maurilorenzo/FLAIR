#!/usr/bin/env Rscript
library(optparse)
option_list <- list( 
	make_option(c("-p", "--p"), default=500, help="number of species"),
	make_option(c("-n", "--n"), default=500, help="number of sampling units"),
	make_option(c("-b", "--backend"), default="c++", help="c++ or tf"),
	make_option("--autograd", default=0, help="use autograd (1) or manual gradients and hessians (0)"),
	make_option("--eager", default=0, help="use eager tf execution (1) or graph (0)"),
	make_option("--bsb", default=512, help="batch size for map_regression_coeffs(...)"),
	make_option("--bse", default=512, help="batch size for map_latent_factors(...)"),
	make_option("--dtype", default=64, help="32 for single precision or 64 double precision")
)
opt <- parse_args(OptionParser(option_list=option_list))

library(truncnorm)
source('FLAIR_wrapper.R')

# DGM params
k <- 10 # number of factors
q <- 10 # number of covariates
p <- opt$p # number of species
n <- opt$n # number of samples
  
sigma_lambda=sqrt(0.5) # sqrt of loadings
sigma_beta=sqrt(0.5) # sqrt of regr coeffs
pi_lambda=0.5 # 1 - sparsity probability for loadings
pi_beta=0.5 # 1 - sparsity probability for regr coeffs

# select c++ or tf MAP computational backend
backend = opt$backend
# generally we want autograd=F, eager_run=F for optimal numerical performance
autograd = opt$autograd
eager_run = opt$eager
batch_size_beta = opt$bsb
batch_size_eta = opt$bse
dtype = opt$dtype
if(backend == "tf"){
	library(reticulate)
	cat("If execution crashes here, check your python+reticulate configuration!\n")
	use_virtualenv("tf")
	source_python("flair_map_tf.py")
	cat("python import successful\n")
}

# generate true params
set.seed(123)
Lambda_0 <- matrix(rtruncnorm(p*k, a=-5, b=5, 0, sigma_lambda)*rbinom(p*k, 1, pi_lambda), ncol=k)
Lambda_0_outer <- Lambda_0 %*% t(Lambda_0)
subsample_index = 1:100
Lambda_0_outer_sub <- Lambda_0_outer[subsample_index, subsample_index]
Beta_0 <- matrix(rtruncnorm(p*q, a=-5, b=5, 0, sigma_beta)*rbinom(p*q, 1, pi_beta), ncol=q)

# generate data
cat("generating data\n")
Eta_0 <- matrix(rnorm(n*k), ncol=k) # latent factors
X <- cbind(rep(1, n), matrix(rnorm(n*(q-1), 0, 1), ncol=q-1)) # covariates (including intercept)
Z_0 <-X %*% t(Beta_0) + Eta_0 %*% t(Lambda_0) # linear predictor
P_0 <- 1/(1+exp(-Z_0)) # probabilities of success
U <- matrix(runif(n*p), ncol=p)
Y <- matrix(0, n, p) # outcomes
Y[U<P_0] = 1

# estimate k
cat("estimating k\n")
k_hat <- select_k(Y, X, observed=matrix(1, n, p), k_max=20, randomized_svd=T)
cat(sprintf("estimated k_hat=%.d\n", k_hat))

# fit FLAIR
cat(sprintf("FLAIR estimation started at: %s\n", date()))
ptm <- proc.time()
flair_estimate <- FLAIR_wrapper(
  Y, X, k_max=20, k=k_hat, method_rho = 'max', eps=0.001, 
  alternate_max=10, max_it=100, tol=0.01, 
  post_process=T, subsample_index = subsample_index, 
  n_MC=300, C_lambda=10,  C_mu=10, C_beta=10, sigma=1.626,
  observed=matrix(1, n, p), randomized_svd=TRUE, loss_tol=0.001,
  backend=backend, autograd=autograd, eager_run=eager_run, 
  batch_size_beta=batch_size_beta, batch_size_eta=batch_size_eta, dtype=dtype)
time_flair = proc.time() - ptm
cat(sprintf("total FLAIR elapsed time: %.1f\n", time_flair[3]))

# compute relative MSE
sum((Lambda_0_outer - flair_estimate$Lambda_outer_mean)^2) / sum((Lambda_0_outer)^2)
sum((Beta_0 - flair_estimate$Beta_tilde)^2) / sum((Beta_0)^2)

# compute coverage
Lambda_outer_qs <- apply(flair_estimate$Lambda_outer_samples_cc, c(1,2),
                         function(x) (quantile(x, probs=c(0.025, 0.975))))
print(mean((Lambda_outer_qs[1,,]<Lambda_0_outer_sub) & (Lambda_outer_qs[2,,]>Lambda_0_outer_sub)))

source('simulations/helpers_simulations.R')
Beta_cc_cis <- compute_ci_Beta(flair_estimate$Beta_tilde[,], flair_estimate$Vs[,1:ncol(X)],
                               flair_estimate$rho_max, alpha=0.05)
print(mean((Beta_cc_cis$ls < Beta_0) & (Beta_cc_cis$us > Beta_0)))
