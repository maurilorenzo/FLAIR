library(dplyr)
library(LaplacesDemon)
library(MASS)
library(matrixStats)
library(rsvd)
library(Rcpp)

sourceCpp('helper_functions.cpp')

post_process_Lambda_samples <- function(Lambda_samples, R){
  S <- dim(Lambda_samples_1$Lambda_samples)[1]
  for(s in 1:S){
    Lambda_samples[s,,] <- t(t(Lambda_samples[s,,]) %*% R)
  }
  return(Lambda_samples)    
}

correct_Beta_samples <- function(Beta_samples, Beta_hat, rho){
  Beta_samples <- rho*sweep(Beta_samples, c(2,3), t(Beta_hat)) 
  Beta_samples <- sweep(Beta_samples, c(2,3), -t(Beta_hat))
  return(Beta_samples)
}

compute_W_j_bar <- function(lambda_j, beta_j, M, X){
  n <- nrow(M) 
  lp <- X %*% beta_j + M %*% lambda_j
  p_j <- 1/(1+exp(-lp))
  res <- p_j * (1 - p_j)
  return(mean(res))
}

compute_svd <- function(Y, k=10, randomized_svd=F){
  n <- nrow(Y); p <- ncol(Y)
  if (n > p) {
    YtY <- t(Y) %*% Y
    if(randomized_svd){s_Y <- rsvd(YtY, k=k, nu=k, nv=k, p = 10, q = 2, sdist = "normal")}
    else{s_Y <- svd(YtY, nv=k, nu=k)}
    V <- s_Y$u[,1:k]
    D <- diag(s_Y$d[1:k]^(1/2))
    U <- Y %*% V %*% diag(s_Y$d[1:k]^(-1/2))
  }
  else {
    if(randomized_svd){s_Y <- rsvd(Y, k=k, nu=k, nv=k, p = 10, q = 2, sdist = "normal")}
    else{s_Y <- svd(Y, nv=k, nu=k)}
    U <- s_Y$u[,1:k]
    D <-diag(s_Y$d[1:k])
    V <- s_Y$v[,1:k]
  }
  return(list(U=U, D=D, V=V))
}

get_variance_params <- function(Beta_hat, Lambda_hat, l_tau_sq_beta=0.5, u_tau_sq_beta=20,
                                l_tau_sq_lambda=0.5, u_tau_sq_lambda=20){
  scales_Beta <- rowMeans(Beta_hat^2)
  l_tau_sq_beta <- 0.5
  u_tau_sq_beta <- 20
  scales_Beta[scales_Beta < l_tau_sq_beta] <- l_tau_sq_beta
  scales_Beta[scales_Beta > u_tau_sq_beta] <- u_tau_sq_beta
  
  scales_Lambda <- rowMeans(Lambda_hat^2)
  l_tau_sq_lambda <- 0.5
  u_tau_sq_lambda <- 20
  scales_Lambda[scales_Lambda < l_tau_sq_lambda] <- l_tau_sq_lambda
  scales_Lambda[scales_Lambda > u_tau_sq_lambda] <- u_tau_sq_lambda
  
  return(list(scales_Beta=scales_Beta, scales_Lambda=scales_Lambda))
}


compute_log_posterior <- function(Y, X, M_hat, Beta_hat, Lambda_hat, observed, scales_Lambda, scales_Beta, alphas_hat=0, tau_alpha=1){
  Z_hat <- X %*% t(Beta_hat) + M_hat %*% t(Lambda_hat)
  log_lik <- sum(-log(1+exp(-Z_hat[observed & Y==1]))) + sum(-log(1+exp(Z_hat[observed & Y==0])))
  log_posterior<- log_lik - 0.5*sum(M_hat^2) - 0.5*sum(diag(scales_Lambda^(-1))%*%Lambda_hat^2) -
    0.5*sum(diag(scales_Beta^(-1))%*%Beta_hat^2) - 0.5*sum(alphas_hat^2)/tau_alpha
  return(log_posterior)
}


alternate_optimization_1_run <- function(
    Y, observed, X, M_hat, Beta_hat, Lambda_hat, scales_Beta, scales_Lambda, max_it, 
    tol, C_lambda, C_mu, C_beta, step_size){
  Beta_Lambda <- map_regression_coeffs_mvlogit(
    Y, observed, X, M_hat, Beta_hat, Lambda_hat, prior_var_beta= scales_Beta, 
    prior_var_lambda=scales_Lambda,  n_MC=1, max_it=max_it, epsilon=tol, C_lambda=C_lambda,
    C_mu=C_mu, C_beta=C_beta, step_size=step_size)
  Lambda_hat <- Beta_Lambda$Lambda_hat
  Beta_hat <- Beta_Lambda$Beta_hat
  M_hat <- map_latent_factors_mvlogit(Y, observed, X, Lambda_hat, Beta_hat, M_hat, 
                                      prior_var=1, max_it=max_it, epsilon=tol)
  return(list(Lambda_hat=Lambda_hat, Beta_hat=Beta_hat, M_hat=M_hat))
}


post_process <- function(X, M_hat, Beta_hat, Lambda_hat){
  n <- nrow(X)
  H_x <- solve(t(X) %*% X) %*% t(X)
  s_M <- svd(M_hat - X %*% H_x %*% M_hat)
  Beta_hat <- Beta_hat + Lambda_hat %*% t(M_hat) %*% t(H_x)
  M_hat <- s_M$u * sqrt(n)
  Lambda_hat <- Lambda_hat %*% s_M$v %*% diag(s_M$d)/sqrt(n)
  return(list(Beta_tilde=Beta_hat, Lambda_tilde=Lambda_hat, M_tilde=M_hat))
}


get_initialization <- function(Y, X, k, randomized_svd=F, eps=0.001,
                               l_tau_sq_beta=0.5, u_tau_sq_beta=20,
                               l_tau_sq_lambda=0.5, u_tau_sq_lambda=20){
  k_tilde <- k + ncol(X)
  s_Y <- compute_svd(Y, k=k_tilde, randomized_svd=randomized_svd)
  Y_hat <- s_Y$U %*% s_Y$D %*% t(s_Y$V)
  Y_hat[(Y_hat>1-eps)] <- 1- eps; Y_hat[(Y_hat<eps)] <- eps
  Z_hat <- logit(Y_hat)
  Beta_hat <- t(Z_hat) %*% X %*% solve(t(X) %*% X)
  Z_hat_c <- Z_hat - X %*% t(Beta_hat)
  n <- nrow(Y)
  alphas_hat <- rep(0, n)
  tau_alpha=1
  s_Y <- compute_svd(Z_hat_c, k=k, randomized_svd=randomized_svd)
  # init M, Lambda
  M_hat <- s_Y$U * sqrt(n)
  Lambda_hat <- s_Y$V %*% s_Y$D / sqrt(n)
  variance_estimates<- get_variance_params(
    Beta_hat, Lambda_hat, l_tau_sq_beta=l_tau_sq_beta, u_tau_sq_beta=u_tau_sq_beta, 
    l_tau_sq_lambda=l_tau_sq_lambda, u_tau_sq_lambda=u_tau_sq_lambda
  )
  scales_Beta <- variance_estimates$scales_Beta
  scales_Lambda <- variance_estimates$scales_Lambda
  return(list(Beta_hat=Beta_hat, Lambda_hat=Lambda_hat, M_hat=M_hat,
              scales_Beta=scales_Beta, scales_Lambda=scales_Lambda))
}

compute_rho <- function(X, M_tilde, Beta_bar, Lambda_bar, subsample_index, method_rho){
  # compute correction factor
  W_bar <- sapply(1:length(subsample_index), function(x) (
    compute_W_j_bar(
      as.matrix(Lambda_bar[subsample_index,])[x,], as.matrix(Beta_bar[subsample_index,])[x,], M_tilde, X))
  )
  B <- compute_B_hessian(Lambda_bar[subsample_index,], W_bar)  
  
  if(is.numeric(method_rho)){
    rho = quantile(B[lower.tri(B, diag=T)], probs=method_rho)
  }
  else if (method_rho=='max'){
    rho = max(B)
  }
  else{
    rho = mean(B[lower.tri(B, diag=T)])
  }
  
  print(paste('rho =', rho))
  
  rho_mean <- mean(B[lower.tri(B, diag=T)])
  rho_max <- max(B[lower.tri(B, diag=T)])
  return(list(rho=rho, rho_mean=rho_mean, rho_max=rho_max))
}


compute_MAP <- function(Y, observed, X, M_hat, Beta_hat, Lambda_hat, scales_Beta, scales_Lambda, max_it=100, 
                        tol=0.01, C_lambda=10, C_mu=10, C_beta=10, step_size=0.5, alternate_max=5, post_process_1=T, 
                        loss_tol=0.001){
  
  log_posterior_old <- compute_log_posterior(Y, X, M_hat, Beta_hat, Lambda_hat, observed, scales_Lambda, scales_Beta)
  print(paste ('initial log posterior: ', log_posterior_old))
  
  for(a in 1:alternate_max){
    
    run_optimizer <- alternate_optimization_1_run(
      Y, observed, X, M_hat, Beta_hat, Lambda_hat, scales_Beta, scales_Lambda, max_it, 
      tol, C_lambda, C_mu, C_beta, step_size=step_size
    )
    Lambda_hat <- run_optimizer$Lambda_hat; Beta_hat <- run_optimizer$Beta_hat; 
    M_hat <- run_optimizer$M_hat; alphas_hat <- run_optimizer$alphas_hat;
    
    log_posterior_new <- compute_log_posterior(Y, X, M_hat, Beta_hat, Lambda_hat, observed,
                                               scales_Lambda, scales_Beta)
    print(paste ('log posterior at iter ', a, log_posterior_new))
    print(paste('% log-posterior change: ', (log_posterior_new - log_posterior_old)/abs(log_posterior_old)))
    
    if(a==1 & post_process_1){
      post_processed_params <- post_process(X, M_hat, Beta_hat, Lambda_hat)
      Beta_hat <- post_processed_params$Beta_tilde; M_hat <- post_processed_params$M_tilde
      Lambda_hat <- post_processed_params$Lambda_tilde
    }
    if((log_posterior_new - log_posterior_old)/abs(log_posterior_old) < loss_tol){
      print('Optimizer has converged!')
      break
    }
    log_posterior_old <- log_posterior_new
  }
  return(list(M_hat=M_hat, Beta_hat=Beta_hat, Lambda_hat=Lambda_hat,
              log_posterior_map = log_posterior_new))
}



logit_log_likelihood <- function(Y, Z, observed){
  ll <- (sum(-log(1+exp(-Z[Y==1 & observed]))) + sum(-log(1+exp(Z[Y==0& observed]))))
  return(ll)
}

compute_jic <- function(Y, Z, k, observed){
  n <- nrow(Y); p <- ncol(Y)
  n_obs <- sum(observed)
  ll <- logit_log_likelihood(Y, Z, observed)
  penalty <- k*max(n, p)*log(n_obs/max(n, p))
  return(-2*ll + penalty)
}

fit_criterion <- function(Y, X, k, U, D, V, observed, eps=0.005, randomized_svd=F){
  Z <- get_linear_predictor(Y, X, k, U, D, V, eps, randomized_svd=randomized_svd)
  jic <- compute_jic(Y, Z, k, observed)
  return(jic)
}

get_linear_predictor<- function(Y, X, k, U, D, V, eps=0.005, approximate=TRUE, randomized_svd=F){
  q <- ncol(X); n <- nrow(Y)
  k_tilde <- q + k
  Y_hat <- as.matrix(U[,1:k_tilde]) %*%  as.matrix(D[1:k_tilde, 1:k_tilde]) %*% t( as.matrix(V[,1:k_tilde]))
  Y_hat[(Y_hat>1-eps)] <- 1- eps
  Y_hat[(Y_hat<eps)] <- eps
  Z_hat <- logit(Y_hat)
  Beta_hat <- t(Z_hat) %*% X %*% solve(t(X) %*% X); Z_hat_c <- Z_hat - X %*% t(Beta_hat)
  s_Z <- compute_svd(Z_hat_c, k=k+1, randomized_svd=randomized_svd)
  U_z <- as.matrix(s_Z$U[,1:k])
  D_z <-as.matrix(s_Z$D[1:k, 1:k])
  V_z <- as.matrix(s_Z$V[,1:k])
  if(approximate==TRUE){ return(U_z %*% D_z %*% t(V_z) +X %*% t(Beta_hat)) }
  scales_Beta <- rowMeans(Beta_hat^2)
  l_tau_sq_beta <- 0.5
  u_tau_sq_beta <- 20
  scales_Beta[scales_Beta < l_tau_sq_beta] <- l_tau_sq_beta
  scales_Beta[scales_Beta > u_tau_sq_beta] <- u_tau_sq_beta
  M_hat <- U_z * sqrt(n)
  Lambda_hat <- V_z %*% D_z / sqrt(n)
  scales_Lambda <- rowMeans(Lambda_hat^2)
  l_tau_sq_lambda <- 0.5
  u_tau_sq_lambda <- 20
  scales_Lambda[scales_Lambda < l_tau_sq_lambda] <- l_tau_sq_lambda
  scales_Lambda[scales_Lambda > u_tau_sq_lambda] <- u_tau_sq_lambda
  C_lambda=10; C_mu=10; C_beta=10
  Beta_Lambda <- map_regression_coeffs_mvlogit(
    Y, X, M_hat, Beta_hat, Lambda_hat, prior_var_beta=scales_Beta, prior_var_lambda=scales_Lambda, 
    n_MC=1, max_it=1000, epsilon=0.001,  C_lambda=C_lambda,  C_mu=C_mu, C_beta=C_beta)
  
  Lambda_hat <- Beta_Lambda$Lambda_hat
  Beta_hat <- Beta_Lambda$Beta_hat
  M_hat <- map_latent_factors_mvlogit(Y, X, Lambda_hat, Beta_hat, M_hat, 
                                      prior_var=1, max_it=1000, epsilon=0.001)
  M_Lambda_hat <- M_hat %*% t(Lambda_hat)
  Z_hat <- X %*% t(Beta_hat) + M_Lambda_hat
  return(Z_hat)
}

svd_Y <- function(Y, X, k_max, randomized_SVD){
  q <- ncol(X)
  k_tilde <- k_max + q 
  if (n > p) {
    YtY <- t(Y) %*% Y
    s_Y <- svd(YtY)
    V <- s_Y$u[,1:k_tilde]
    D <- diag(s_Y$d[1:k_tilde]^(1/2))
    U <- Y %*% V %*% diag(s_Y$d[1:k_tilde]^(-1/2))
  }
  else {
    s_Y <- svd(Y, nv=k_tilde, nu=k_tilde)
    U <- s_Y$u[,1:k_tilde]
    D <-(s_Y$d[1:k_tilde])
    V <- s_Y$v[,1:k_tilde]
  }
  return(list(U=U, D=D, V=V))
}  

select_k <- function(Y, X, observed, k_max=20, eps=0.005, randomized_svd=F){
  s_Y <- compute_svd(Y, k_max + ncol(X), randomized_svd)
  jics <- rep(0, k_max)
  jics <- sapply(1:(k_max), function(x) (fit_criterion(Y, X, x, s_Y$U, s_Y$D, s_Y$V, observed, eps=0.001, randomized_svd)))
  print(jics)
  k_optimal <- which.min(jics)
  plot(1:k_max, jics, type='l', xlab='k', ylab='JIC'); points(k_optimal, jics[k_optimal], col='red')
  print(paste('k = ', k_optimal, '; JIC = ', jics[k_optimal] ))
  return(k_optimal)
}

compute_ci_Beta <- function(Beta_hat, Vs_hat, rho, alpha=0.05){
  alpha_2 = alpha/2
  q <- qnorm(1-alpha_2)
  qs <- q*rho*sqrt(Vs_hat);
  ls <- Beta_hat - qs; us <- Beta_hat + qs
  return(list(ls=ls, us=us))
} 


FLAIR_wrapper <- function(
    Y, X, k_max=20, k=NA, method_rho='max', eps=0.005,  alternate_max=3, max_it=1000, tol=0.001,
    post_process=T,  subsample_index = 1:100, n_MC=1000, C_lambda=10,  C_mu=10, C_beta=10, sigma=1.702, 
    observed = NA, randomized_svd=F, step_size_newton=1, loss_tol=0.001,
    provide_posterior_samples=T, backend="c++", autograd=TRUE, 
    batch_size_beta=512, batch_size_eta=512, eager_run=FALSE, retrace_print=FALSE, dtype=dtype) {
  
  if(is.na(sum(observed))) {observed = matrix(1, nrow=nrow(Y), ncol=ncol(Y) )}
  p <- ncol(Y); n<-nrow(Y); q <- ncol(X)
  if(sum(observed)< n*p){
    Y <- Y %>% 
      as.data.frame() %>% 
      mutate_all(~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x))  %>%
      as.matrix()
  }
  
  # select k
  if(is.na(k)){
    print('choosing k via JIC')
    k <- select_k(Y, X, observed, k_max=k_max)
  }
  
  # compute joint MAP  
  ## init
  init_values <- get_initialization(Y, X, k, randomized_svd=randomized_svd, eps=eps, 
                                    l_tau_sq_beta=0.5, u_tau_sq_beta=20, l_tau_sq_lambda=0.5, 
                                    u_tau_sq_lambda=20)
  M_hat <- init_values$M_hat; Lambda_hat <- init_values$Lambda_hat; Beta_hat <- init_values$Beta_hat
  scales_Lambda <- init_values$scales_Lambda; scales_Beta <- init_values$scales_Beta
  
  Beta_hat[,1][Beta_hat[,1]>C_mu] <- C_mu;   Beta_hat[,1][Beta_hat[,1]< -C_mu] <- (-C_mu)
  Beta_hat[,-1][Beta_hat[,-1]>C_beta] <- C_beta;   Beta_hat[,-1][Beta_hat[,-1]< -C_beta] <- (-C_beta)
  Lambda_hat[Lambda_hat>C_lambda] <- C_lambda;   Lambda_hat[Lambda_hat< -C_lambda] <- (-C_lambda)
  M_hat[M_hat>2 *sqrt(log(k*n))] <- 2 *sqrt(log(k*n));   M_hat[M_hat< -  2*sqrt(log(k*n))] <- (-2 *sqrt(log(k*n)));
  
  start_time = proc.time()
  if(backend == "c++"){
    cat("using C++ backend for MAP estimate\n")
    MAP_estimate <- compute_MAP(Y, observed, X, M_hat, Beta_hat, Lambda_hat, scales_Beta, scales_Lambda, max_it, 
                                tol, C_lambda, C_mu, C_beta, step_size_newton, alternate_max, 
                                post_process_1=FALSE, loss_tol=loss_tol)
    M_tilde <- MAP_estimate$M_hat; 
    Beta_init <- MAP_estimate$Beta_hat;
    Lambda_init <- MAP_estimate$Lambda_hat
    log_posterior_map <- MAP_estimate$log_posterior_map
  } else{
    cat("using TensorFlow backend for MAP estimate\n")
    Y_tf = Y
    Y_tf[!observed] = NA
    scales_Beta_tf = matrix(scales_Beta, ncol=1)
    scales_Lambda_tf = matrix(scales_Lambda, ncol=1)
    scales_Eta_tf = matrix(1, nrow(Y), 1)
    MAP_estimate <- compute_MAP_tf(Y_tf, X, Beta_hat, M_hat, Lambda_hat, scales_Beta_tf, scales_Eta_tf, scales_Lambda_tf, 
                                   as.integer(max_it), tol, C_lambda, C_mu, C_beta, step_size_newton, as.integer(alternate_max), 
                                   post_process_1=FALSE, loss_tol=loss_tol, autograd=autograd,
                                   batch_size_beta=as.integer(batch_size_beta), batch_size_eta=as.integer(batch_size_eta),
                                   eager_run=eager_run, retrace_print=retrace_print, dtype=dtype)
    Beta_init <- MAP_estimate[[1]]
    M_tilde <- MAP_estimate[[2]]
    Lambda_init <- MAP_estimate[[3]]
    log_posterior_map <- MAP_estimate[[4]]
  }
  map_time = proc.time() - start_time
  cat(sprintf("FLAIR MAP estimate elapsed time: %.1f\n", map_time[3]))
  
  if(post_process){
    post_processed_params <- post_process(X, M_tilde, Beta_init, Lambda_init)
    Beta_tilde <- post_processed_params$Beta_tilde; Beta_init  = Beta_tilde
    M_tilde <- post_processed_params$M_tilde; alphas_tilde <- post_processed_params$alphas_tilde
    M_tilde_max <- max(abs(M_tilde));
    if(M_tilde_max>2 *sqrt(log(k*n))){C_lambda = C_lambda * 2 *sqrt(log(k*n)) / M_tilde_max}
    Lambda_tilde = post_processed_params$Lambda_tilde; Lambda_init = Lambda_tilde
    Beta_init[,1][Beta_init[,1]>C_mu] <- C_mu;   Beta_init[,1][Beta_init[,1]< -C_mu] <- (-C_mu)
    Beta_init[,-1][Beta_init[,-1]>C_beta] <- C_beta;   Beta_init[,-1][Beta_init[,-1]< -C_beta] <- (-C_beta)
    Lambda_init[Lambda_init>C_lambda] <- C_lambda;   Lambda_init[Lambda_init< -C_lambda] <- (-C_lambda)
  }
  
  print('post-processed jMAP computed')
  
  rhos <- compute_rho(X, M_tilde, Beta_tilde, Lambda_tilde, subsample_index, method_rho)
  rho <- rhos$rho; rho_mean <- rhos$rho_mean; rho_max <- rhos$rho_max
  rho <- rho_max
  print(rho)
  
  
  Lambda_outer <- Lambda_tilde %*% t(Lambda_tilde)
  output <- list(
    'Beta_tilde' = Beta_tilde, 
    'Lambda_tilde' = Lambda_tilde, 'M_tilde'=M_tilde, 'Beta_tilde'=Beta_tilde, 'Lambda_tilde'=Lambda_tilde, 'D'=D
  )
  
  if(! provide_posterior_samples){
    Vs <- compute_variances(
      Y, observed, X, M_tilde, Beta_tilde, Lambda_tilde, scales_Beta, scales_Lambda,
      n_MC=n_MC
    )
    D <- colSums(Vs[,-(1:q)])
    Lambda_outer_mean <- Lambda_outer + diag(as.vector(D))
    Lambda_outer_mean_cc <- Lambda_outer + rho_max^2*diag(as.vector(D))
    output$Lambda_outer_mean <- Lambda_outer_mean; output$Lambda_outer_mean_cc <- Lambda_outer_mean_cc
    
  }
  
  if(! provide_posterior_samples){
    return(output)
  }
  
  Vs <- compute_variances(
    Y, observed, X, M_tilde, Beta_tilde, Lambda_tilde, scales_Beta, scales_Lambda,
    n_MC=n_MC
  )
  
  
  Beta_Lambda <- sample_full_conditional_logit(
    Y, observed, X, M_tilde, Beta_tilde, Lambda_tilde, scales_Beta, scales_Lambda, n_MC=n_MC
  )
  
  D <- Beta_Lambda$D
  Lambda_outer_mean <- Lambda_outer + diag(as.vector(D))
  Lambda_outer_mean_cc <- Lambda_outer + rho_max^2*diag(as.vector(D))
  output$Lambda_outer_mean <- Lambda_outer_mean; output$Lambda_outer_mean_cc <- Lambda_outer_mean_cc
  
  Lambda_tilde_sub <- as.matrix(Lambda_tilde[subsample_index,]); Beta_tilde_sub <- as.matrix(Beta_tilde[subsample_index,])
  Lambda_outer_samples <- sample_Lambda_outer(Beta_Lambda$Lambda_samples[,,subsample_index],
                                              Lambda_tilde_sub, rho)
  #Beta_samples_cc <- correct_Beta_samples(Beta_Lambda$Beta_samples[,,], 
  #                                        Beta_tilde, rho)
  
  output$Lambda_outer_samples = Lambda_outer_samples$Lambda_outer_samples; output$Lambda_outer_samples_cc = Lambda_outer_samples$Lambda_outer_samples_cc
  output$Lambda_tilde_sub; 
  output$Lambda_samples = Beta_Lambda$Lambda_samples;
  #output$Beta_samples = Beta_Lambda$Beta_samples
  output$D = D; output$beta_tilde_sub = Beta_tilde_sub; 
  #output$Beta_samples_cc = Beta_samples_cc; 
  output$rho_mean = rho_mean; output$rho_max = rho_max
  output$Vs = Vs; output$var_Beta = scales_Beta; output$var_Lambda = scales_Lambda
  return(output)
}






