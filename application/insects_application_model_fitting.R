#!/usr/bin/env Rscript
library(optparse)
option_list <- list( 
	make_option(c("-b", "--backend"), default="c++", help="c++ or tf"),
	make_option("--autograd", default=0, help="use autograd (1) or manual gradients and hessians (0)"),
	make_option("--eager", default=0, help="use eager tf execution (1) or graph (0)"),
	make_option("--bsb", default=512, help="batch size for map_regression_coeffs(...)"),
	make_option("--bse", default=512, help="batch size for map_latent_factors(...)"),
	make_option("--dtype", default=64, help="32 for single precision or 64 double precision")
)
opt <- parse_args(OptionParser(option_list=option_list))

source("../FLAIR_wrapper.R", chdir=TRUE)
source('helpers_application.R')

backend = opt$backend
autograd = opt$autograd
eager_run = opt$eager
batch_size_beta = opt$bsb
batch_size_eta = opt$bse
dtype = opt$dtype
if(backend == "tf"){
	library(reticulate)
	cat("If execution crashes here, check your python+reticulate configuration!\n")
	use_virtualenv("tf") # change to your proper python virtual environment name or replace with use_python(...)
	source_python("../flair_map_tf.py")
	cat("python import successful\n")
}

# - DATA CLEANING ####

load('allData.RData') # include the data used
da = as.Date(meta$COLL_DATE)
jday =  julian(da)-julian(as.Date("2021-01-01"))
co = cos(2*pi*jday/365)
si = sin(2*pi*jday/365)
co2 = cos(4*pi*jday/365)
si2 = sin(4*pi*jday/365)
XData = data.frame(seqdepth = log(meta$readCount), temp = meta$temp, prec = meta$prec, 
                   temp2 = meta$temp^2, prec2 = meta$prec^2, temp_prec = meta$temp*meta$prec,
                   co = co, si = si, co2 = co2, si2 = si2)

scales_x <- colSds(as.matrix(XData))
means_x <- colMeans(XData) 
XData = cbind(rep(1, nrow(Y)) , scale(XData))
dim(XData)
XSample = model.matrix(~-1+as.factor(meta$SITE))
X <- XData


# -PREDICTIVE ACCURACY EXPERIMENTS ####

## -Th=15 #####
### - Data Splitting ####
train_test_val_split_th15 <- train_test_val_split(Y, th=15)
observed_th15 <- train_test_val_split_th15$observed
Y_sel_th15 <- train_test_val_split_th15$Y_sel
Y_train_th15 <- train_test_val_split_th15$Y_train
test_mask_th15 <- train_test_val_split_th15$test_mask
val_mask_th15 <- train_test_val_split_th15$val_mask
Y_train_imputed_th15 <- train_test_val_split_th15$Y_train_imputed
test_set_th15 <- train_test_val_split_th15$test_set
val_set_th15 <- train_test_val_split_th15$val_set
train_set_th15 <- train_test_val_split_th15$train_set

set.seed(123)
# k_insects <- select_k(Y_sel_th15, X, observed=observed_th15, k_max=20, randomized_svd=T) #
k_insects <- 7

### - FLAIR ####
set.seed(123)
cat(sprintf("FLAIR th15 estimation started at: %s\n", date()))
ptm <- proc.time()
flair.insects.th15 <- FLAIR_wrapper(
  Y_train_imputed_th15, X, k_max=k_insects+2, k=k_insects, method_rho = 'max', eps=0.005, 
  alternate_max=10, max_it=100, tol=0.01, post_process=T, subsample_index = 1:5, n_MC=100,
  C_lambda=10,  C_mu=10, C_beta=10, sigma=1.626, observed=observed_th15, randomized_svd=T, 
  step_size_newton = 0.3, loss_tol=0.001, backend=backend, eager_run=eager_run, autograd=autograd, 
  batch_size_beta=batch_size_beta, batch_size_eta=batch_size_eta, dtype=dtype)
time.flair.insects <- proc.time() - ptm; 
cat(sprintf("Elapsed time th15, FLAIR: %.1f\n", time.flair.insects[3]))

auc_flair_th15 <- compute_auc(Y_sel_th15, X, flair.insects.th15$Beta_tilde,
                              flair.insects.th15$M_tilde, flair.insects.th15$Lambda_tilde,
                              test_set_th15, val_set_th15, train_set_th15)


### - GMF  ####
library("gmf")

# with the best configuration
# NEWTON
set.seed(123)
cat(sprintf("GMF-newton th15 estimation started at: %s\n", date()))
ptm <- proc.time()
model.gmf.newton.insects <- gmf(Y = Y_train_th15[,], X =X[,-1], p = k_insects,
                                gamma=0.2, maxIter = 1000,
                                family = binomial(),
                                method = "quasi",
                                penaltyV = 1,
                                penaltyU = 1,
                                penaltyBeta= 10,
                                intercept = T,
                                tol = 0.001,
                                init='svd');
time.gmf.newton <- proc.time() - ptm;
cat(sprintf("Elapsed time th15, gmf.newton: %.1f\n", time.gmf.newton[3]))

auc_gmf_newton_th15 <- compute_auc(Y_sel_th15, X, t(model.gmf.newton.insects$beta),
                                   model.gmf.newton.insects$u, model.gmf.newton.insects$v,
                                   test_set_th15, val_set_th15, train_set_th15)

# AIRWLS
set.seed(123)
cat(sprintf("GMF-airwls th15 estimation started at: %s\n", date()))
ptm <- proc.time()
model.gmf.airwls.insects <- gmf(Y = Y_train_th15[,], X =X[,-1], p = k_insects,
                                gamma=0.2, maxIter = 1000,
                                family = binomial(),
                                method = "airwls",
                                penaltyV = 0.2,
                                penaltyU = 1,
                                penaltyBeta= 0.2,
                                intercept = T,
                                tol = 0.001);
time.gmf.airwls <- proc.time() - ptm;
cat(sprintf("Elapsed time th15, gmf.airwls: %.1f\n", time.gmf.airwls[3]))

auc_gmf_airwls_th15 <- compute_auc(Y_sel_th15, X, t(model.gmf.airwls.insects$beta),
                                   model.gmf.airwls.insects$u, model.gmf.airwls.insects$v,
                                   test_set_th15, val_set_th15, train_set_th15)

## -TH=3 #####
### - Data Splitting ####
train_test_val_split_th3 <- train_test_val_split(Y, th=3)
observed_th3 <- train_test_val_split_th3$observed
Y_sel_th3 <- train_test_val_split_th3$Y_sel
Y_train_th3 <- train_test_val_split_th3$Y_train
test_mask_th3 <- train_test_val_split_th3$test_mask
val_mask_th3 <- train_test_val_split_th3$val_mask
Y_train_imputed_th3 <- train_test_val_split_th3$Y_train_imputed
test_set_th3 <- train_test_val_split_th3$test_set
val_set_th3 <- train_test_val_split_th3$val_set
train_set_th3 <- train_test_val_split_th3$train_set

### - FLAIR ####
set.seed(123)
cat(sprintf("FLAIR th3 estimation started at: %s\n", date()))
ptm <- proc.time()
flair.insects.th3 <- FLAIR_wrapper(
  Y_train_imputed_th3, X, k_max=k_insects+2, k=k_insects, method_rho = 'max', eps=0.005, 
  alternate_max=10, max_it=100, tol=0.01, post_process=T, subsample_index = 1:5, n_MC=100,
  C_lambda=10,  C_mu=10, C_beta=10, sigma=1.626, observed=observed_th3, randomized_svd=T, 
  step_size_newton = 0.25, loss_tol=0.001, backend=backend, eager_run=eager_run, autograd=autograd, 
  batch_size_beta=batch_size_beta, batch_size_eta=batch_size_eta, dtype=dtype)
time.flair.insects <- proc.time() - ptm; 
cat(sprintf("Elapsed time th3, FLAIR: %.1f\n", time.flair.insects[3]))

auc_flair_th3 <- compute_auc(Y_sel_th3, X, flair.insects.th3$Beta_tilde,
                              flair.insects.th3$M_tilde, flair.insects.th3$Lambda_tilde,
                              test_set_th3, val_set_th3, train_set_th3)



### - GMF  ####
# NEWTON
set.seed(123)
cat(sprintf("GMF-newton th3 estimation started at: %s\n", date()))
ptm <- proc.time()
model.gmf.newton.insects.th3 <- gmf(Y = Y_train_th3[,], X =X[,-1], p = 7,
                                    gamma=0.2, maxIter = 1000,
                                    family = binomial(),
                                    method = "quasi",
                                    penaltyV = 20,
                                    penaltyU = 1,
                                    penaltyBeta= 20,
                                    intercept = T,
                                    tol = 0.001,
                                    init='svd');
time.gmf.newton <- proc.time() - ptm;
cat(sprintf("Elapsed time th3, gmf.newton: %.1f\n", time.gmf.newton[3]))

auc_gmf_newton_th3 <- compute_auc(Y_sel_th3, X, t(model.gmf.newton.insects$beta),
                                   model.gmf.newton.insects$u, model.gmf.newton.insects$v,
                                   test_set_th3, val_set_th3, train_set_th3)


# INFERENCE ON FULL DATA (th=15) ####
n <- nrow(Y.sel.th15); p <- ncol(Y.sel.th15); c(n, p)
species_names <- c(taxonomy$species[sel.sp])
taxonomy$family[sel.sp][w_species]
w_species <- which(!grepl("^pseudo", as.character(species_names[1:p])))
set.seed(123)
subsample_index <- w_species
subsample_index <- c(1:1000, w_species)
ptm <- proc.time();
set.seed(123)
flair.insects.th15.full <- FLAIR_wrapper(
  Y.sel.th15, X, k_max=10, k=7, method_rho = 'max', eps=0.005, alternate_max=10,
  max_it=100, tol=0.01, post_process=T, subsample_index = subsample_index,
  n_MC=100, C_lambda=10,  C_mu=10, C_beta=10, sigma=1.702,
  observed=matrix(1, n, p), randomized_svd=T, step_size_newton = 0.25,
  loss_tol=0.001);

time.flair.insects <- proc.time() - ptm; time.flair.insects[3]
time.flair.insects[3]
time.flair.insects[3]/60

set.seed(123)
subsample_index <- c(sample(1:p, 1000), w_species)
flair.insects.th15.full.samples.random <- sample_full_conditional_logit(
  Y.sel.th15[,subsample_index], matrix(1, n, length(subsample_index)),
  X, flair.insects.th15.full$M_tilde,
  flair.insects.th15.full$Beta_tilde[subsample_index,],
  flair.insects.th15.full$Lambda_tilde[subsample_index,],
  flair.insects.th15.full$var_Beta[subsample_index],
  flair.insects.th15.full$var_Lambda[subsample_index],
  n_MC=200, rho = flair.insects.th15.full$rho_max
)