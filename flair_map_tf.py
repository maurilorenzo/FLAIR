import numpy as np
import setuptools.dist
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from time import time
tfla, tfm, tfr = tf.linalg, tf.math, tf.random
tfd = tfp.distributions


@tf.function
def compute_log_posterior_tf(Y, X, Beta, Eta, Lambda, prior_var_beta=1, prior_var_eta=1, prior_var_lambda=1, 
                          retrace_print=True, dtype=np.float64):
  if retrace_print == True: print("retracing compute_log_posterior")
  Yo = tf.cast(~tfm.is_nan(tf.cast(Y, dtype)), dtype)
  Z = tf.matmul(X, Beta, transpose_b=True) + tf.matmul(Eta, Lambda, transpose_b=True)
  obsDist = tfd.Bernoulli(logits=Z)
  logLikeY = tfm.multiply_no_nan(obsDist.log_prob(Y), Yo)
  logLike = tf.reduce_sum(logLikeY)
  logPriorBeta = -0.5*tf.reduce_sum(prior_var_beta**-1 * Beta**2)
  logPriorEta = -0.5*tf.reduce_sum(prior_var_eta**-1 * Eta**2)
  logPriorLambda = -0.5*tf.reduce_sum(prior_var_lambda**-1 * Lambda**2)
  logPost = logLike + logPriorEta + logPriorLambda + logPriorBeta
  return logPost


@tf.function
def map_regression_coeffs(Y, X, Beta, Eta, Lambda, prior_var_beta, prior_var_lambda, max_it, epsilon=0.001,
                          C_mu=10, C_beta=10, C_lambda=10, step_size=1, autograd=True, batch_size=512, 
                          retrace_print=True, dtype=np.float64):
  if retrace_print == True: print("retracing map_regression_coeffs")
  Yo = tf.cast(~tfm.is_nan(tf.cast(Y, float)), dtype)
  BetaLambda = tf.concat([Beta, Lambda], -1)
  C_comb = tf.concat([C_mu*tf.ones([1],dtype), C_beta*tf.ones([Beta.shape[1]-1],dtype), C_lambda*tf.ones(Lambda.shape[1],dtype)], -1)
  XEta = tf.concat([X, Eta], -1)
  prior_var_beta_lambda = tf.concat([tf.broadcast_to(prior_var_beta, Beta.shape), tf.broadcast_to(prior_var_lambda, Lambda.shape)], -1)
  if autograd == True:
    log_post_beta_lambda = lambda BetaLambda: compute_log_posterior_tf(Y, X, BetaLambda[:,:Beta.shape[1]], Eta, BetaLambda[:,Beta.shape[1]:],
                                                                       prior_var_beta=prior_var_beta, prior_var_lambda=prior_var_lambda, 
                                                                       retrace_print=retrace_print, dtype=dtype)
  else:
    batch_n = np.ceil(Y.shape[0] / batch_size)
  
  for i in tf.range(max_it):
    if autograd == True:
      with tf.GradientTape() as tape2:
        tape2.watch(BetaLambda)
        with tf.GradientTape() as tape1:
          tape1.watch(BetaLambda)
          log_post = log_post_beta_lambda(BetaLambda)
        gradient = tape1.gradient(log_post, BetaLambda)
      batch_hessian = tape2.batch_jacobian(gradient, BetaLambda)
    else:
      Z = tf.matmul(XEta, BetaLambda, transpose_b=True)
      F = tfm.sigmoid(Z)
      S = tf.cast(Y, dtype) - F
      gradient = tf.matmul(tfm.multiply_no_nan(S, Yo), XEta, transpose_a=True) - BetaLambda / prior_var_beta_lambda
      batch_hessian = -tfla.diag(prior_var_beta_lambda**-1)
      F1mFobs = tfm.multiply_no_nan(F*(1-F), Yo)
      for b in tf.range(batch_n):
        XEta_batch = tf.gather(XEta, tf.range(b*batch_size, tfm.minimum((b+1)*batch_size, Y.shape[0]), dtype=np.int32))
        F1mFobs_batch = tf.gather(F1mFobs, tf.range(b*batch_size, tfm.minimum((b+1)*batch_size, Y.shape[0]), dtype=np.int32))
        batch_hessian -= tf.einsum("ih,ij,ig->jhg", XEta_batch, F1mFobs_batch, XEta_batch)
    update = tf.squeeze(step_size * tfla.solve(batch_hessian, gradient[:,:,None]), -1)
    BetaLambda -= update
    BetaLambda = tf.clip_by_value(BetaLambda, -C_comb, C_comb)
    if tf.reduce_max(tf.norm(update, 2, axis=1)) < epsilon:
      break
  # tf.print("map_regression_coeffs - iterations:", i)
  Beta, Lambda = tf.split(BetaLambda, [Beta.shape[-1], Lambda.shape[-1]], axis=-1)
  return Beta, Lambda
  

@tf.function
def map_latent_factors(Y, X, Beta, Eta, Lambda, prior_var_eta, max_it, epsilon=0.01, step_size=1, 
                       autograd=True, batch_size=512, retrace_print=False, dtype=np.float64):
  if retrace_print == True: print("retracing map_latent_factors")
  C_m = 2 * np.sqrt(np.log(Y.shape[0]*Lambda.shape[1]))
  Yo = tf.cast(~tfm.is_nan(tf.cast(Y, float)), dtype)
  Eta = tf.convert_to_tensor(Eta)
  if autograd == True:
    log_post_eta = lambda Eta: compute_log_posterior_tf(Y, X, Beta, Eta, Lambda, prior_var_eta=prior_var_eta, retrace_print=retrace_print, dtype=dtype)
  
  for i in tf.range(max_it):
    if autograd == True:
      with tf.GradientTape() as tape2:
        tape2.watch(Eta)
        with tf.GradientTape() as tape1:
          tape1.watch(Eta)
          log_post = log_post_eta(Eta)
        gradient = tape1.gradient(log_post, Eta)
      batch_hessian = tape2.batch_jacobian(gradient, Eta)
    else:
      batch_n = np.ceil(Y.shape[1] / batch_size)
      Z = tf.matmul(X, Beta, transpose_b=True) + tf.matmul(Eta, Lambda, transpose_b=True)
      F = tfm.sigmoid(Z)
      S = tf.cast(Y, dtype) - F
      gradient = tf.matmul(tfm.multiply_no_nan(S, Yo), Lambda) - Eta / prior_var_eta
      F1mFobs = tfm.multiply_no_nan(F*(1-F), Yo)
      batch_hessian = -(prior_var_eta**-1)[:,:,None] * tf.eye(Lambda.shape[1], dtype=dtype)
      for b in tf.range(batch_n):
        Lambda_batch = tf.gather(Lambda, tf.range(b*batch_size, tfm.minimum((b+1)*batch_size, Y.shape[1]), dtype=np.int32), axis=0)
        F1mFobs_batch = tf.gather(F1mFobs, tf.range(b*batch_size, tfm.minimum((b+1)*batch_size, Y.shape[1]), dtype=np.int32), axis=1)
        batch_hessian -= tf.einsum("jh,ij,jg->ihg", Lambda_batch, F1mFobs_batch, Lambda_batch) 
    update = tf.squeeze(step_size * tfla.solve(batch_hessian, gradient[:,:,None]), -1)
    Eta -= update
    Eta = tf.clip_by_value(Eta, -C_m, C_m)
    if tf.reduce_max(tf.norm(update, 2, axis=1)) < epsilon:
      break
  # tf.print("map_latent_factors - iterations:", i)
  return Eta
  

def compute_MAP_tf(Y, X, Beta, Eta, Lambda, prior_var_beta, prior_var_eta, prior_var_lambda, max_it=100, tol=0.01, 
                C_lambda=10, C_mu=10, C_beta=10, step_size=0.5, alternate_max=5, post_process_1=True, loss_tol=0.001, 
                autograd=False, eager_run=False, batch_size_beta=512, batch_size_eta=512, retrace_print=False, dtype=np.float64):
  if dtype == 32:
    dtype = np.float32
  elif dtype == 64:
    dtype = np.float64
  elif type(dtype) != type:
    raise Exception("dtype argument must be a valid type or 32 or 64")
  
  tf.config.run_functions_eagerly(eager_run)
  if eager_run == True: print("running MAP estimation eagerly")
  v_list = [X, Beta, Eta, Lambda, prior_var_beta, prior_var_eta, prior_var_lambda]
  X, Beta, Eta, Lambda, prior_var_beta, prior_var_eta, prior_var_lambda = [tf.cast(v, dtype) for v in v_list]
  log_posterior_old = compute_log_posterior_tf(Y, X, Beta, Eta, Lambda, prior_var_beta, prior_var_eta, prior_var_lambda, retrace_print, dtype)
  print(f'initial log posterior: {log_posterior_old}')
  for a in range(alternate_max):
    Beta, Lambda = map_regression_coeffs(Y, X, Beta, Eta, Lambda, prior_var_beta, prior_var_lambda, max_it, 
                                         epsilon=tol, C_mu=C_mu, C_beta=C_beta, C_lambda=C_lambda, step_size=step_size, 
                                         autograd=autograd, batch_size=batch_size_eta, retrace_print=retrace_print, dtype=dtype)
    Eta = map_latent_factors(Y, X, Beta, Eta, Lambda, prior_var_eta=prior_var_eta, max_it=max_it, epsilon=tol, 
                             autograd=autograd, batch_size=batch_size_eta, retrace_print=retrace_print, dtype=dtype)
    log_posterior_new = compute_log_posterior_tf(Y, X, Beta, Eta, Lambda, prior_var_beta, prior_var_eta, prior_var_lambda, retrace_print, dtype)
    print(f'log posterior at iter {a+1}: {log_posterior_new}')
    log_posterior_delta_rel = (log_posterior_new - log_posterior_old) / tfm.abs(log_posterior_old)
    print(f'% log-posterior change: {log_posterior_delta_rel}')
    log_posterior_old = log_posterior_new
    
    if log_posterior_delta_rel < loss_tol:
      print('Optimizer has converged!', flush=True)
      break
    
  return Beta.numpy(), Eta.numpy(), Lambda.numpy(), log_posterior_new.numpy()


def main(autograd=False, eager_run=False, enable_plotting=False, dtype=np.float64):
  np.random.seed(42)
  tfr.set_seed(42)
  p = 21
  n = 27
  q = 11
  k = 13
  max_it = 10
  retrace_print = not eager_run
  tf.config.run_functions_eagerly(eager_run)
    
  X = np.random.normal(size=[n,q])
  Eta = np.random.normal(size=[n,k])
  Beta = np.random.normal(size=[p,q])
  Lambda = np.random.normal(size=[p,k])
  L = np.matmul(X, Beta.T) + np.matmul(Eta, Lambda.T)
  Y = tfd.Bernoulli(L).sample().numpy()
  prior_var_beta = np.ones([p,1])
  prior_var_lambda = np.ones([p,1])
  prior_var_eta = np.ones([n,1])
  
  v_list = [X, Eta, Beta, Lambda, prior_var_beta, prior_var_lambda, prior_var_eta]
  X, Eta, Beta, Lambda, prior_var_beta, prior_var_lambda, prior_var_eta = [v.astype(dtype) for v in v_list]
  BetaGen, LambdaGen, EtaGen = Beta, Lambda, Eta
  lp = compute_log_posterior_tf(Y, X, Beta, Eta, Lambda, prior_var_beta, prior_var_eta, prior_var_lambda, retrace_print, dtype)
  print("Likelihood test", lp.numpy(), flush=True)  
  BetaMap, LambdaMap = map_regression_coeffs(Y, X, 0*Beta, EtaGen, 0*Lambda, prior_var_beta, prior_var_lambda, max_it,
                                             autograd=autograd, retrace_print=retrace_print, dtype=dtype)
  
  if enable_plotting:
    plt.subplot(1,2,1)
    plt.scatter(BetaGen, BetaMap.numpy())
    plt.gca().axline([0,0], [1,1], color="black", linestyle="dashed")
    plt.title("Beta")
    plt.subplot(1,2,2)
    plt.scatter(LambdaGen, LambdaMap.numpy())
    plt.gca().axline([0,0], [1,1], color="black", linestyle="dashed")
    plt.title("Lambda")
    plt.show()
  
  EtaMap = map_latent_factors(Y, X, BetaGen, 0*Eta, LambdaGen, prior_var_eta, max_it, 
                              autograd=autograd, retrace_print=retrace_print, dtype=dtype)
  if enable_plotting:
    plt.scatter(EtaGen, EtaMap.numpy())
    plt.gca().axline([0,0], [1,1], color="black", linestyle="dashed")
    plt.title("Eta")
    plt.show()
  
  multGen, multRand = 0.1, 0.3
  BetaInit = multGen*BetaGen + multRand*np.random.normal(size=BetaGen.shape)
  EtaInit = multGen*EtaGen + multRand*np.random.normal(size=EtaGen.shape)
  LambdaInit = multGen*LambdaGen + multRand*np.random.normal(size=LambdaGen.shape)
  startTime = time()
  BetaInit, EtaInit, LambdaInit = [v.astype(dtype) for v in [BetaInit, EtaInit, LambdaInit]]
  print("MAP test started", flush=True)
  res = compute_MAP_tf(Y, X, BetaInit, EtaInit, LambdaInit, prior_var_beta, prior_var_eta, prior_var_lambda, max_it=10, tol=0.01, 
                    C_lambda=10, C_mu=10, C_beta=10, step_size=0.1, alternate_max=2, post_process_1=True, loss_tol=0.001,
                    autograd=autograd, eager_run=eager_run, retrace_print=retrace_print, dtype=dtype)
  stopTime = time()
  print(f"elapsed time: {stopTime-startTime:.1f}")

if __name__ == "__main__":
    main()