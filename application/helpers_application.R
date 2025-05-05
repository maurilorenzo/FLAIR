library(pROC)




train_test_split_j <- function(y){
  n <- length(y);
  pos1 <- which(y==1)
  n1 <- length(pos1); random_index1 <- sample(1:n1, n1)
  #n_train1 <- min(floor(0.8*n1), n1 -2)
  n_train1 <- floor(0.8*n1)
  train_random_index1 <- random_index1[1:n_train1]; test_index1 <- pos1[-train_random_index1]
  train_index1 <- pos1[train_random_index1]; 
  l_test1 <- length(test_index1)
  if(l_test1>1){  
    validation_index1 <- test_index1[(as.integer(l_test1/2)+1):l_test1]; test_index1 <- test_index1[1:as.integer(l_test1/2)]; 
  }
  else if (runif(1)< 0.5){test_index1 <- test_index1; validation_index1 <- c();}
  else{validation_index1 <- test_index1; test_index1 <- c();}
  pos0 <- which(y==0)
  n0 <- length(pos0); random_index0 <- sample(1:n0, n0)
  #train_random_index0 <- random_index0[0:floor(0.8*n0)]
  train_random_index0 <- random_index0[1:floor(0.8*n0)]; test_index0 <- pos0[-train_random_index0]
  train_index0 <- pos0[train_random_index0]; 
  l_test0 <- length(test_index0)
  if(l_test0>1){  
    validation_index0 <- test_index0[(as.integer(l_test0/2)+1):l_test0]; test_index0 <- test_index0[1:as.integer(l_test0/2)]; 
  }
  else if (runif(1)< 0.5){test_index0 <- test_index0; validation_index0 <- c()}
  else{validation_index0 <- test_index0; test_index0 <- c();}
  
  return(list(train=c(train_index1, train_index0), test=c(test_index1, test_index0), val=c(validation_index1, validation_index0)))
}


train_test_val_split <- function(Y, th=15){
  prev = colSums(Y)
  sel.sp = prev>=th
  Y.sel = Y[,sel.sp]
  n <- nrow(Y.sel); p <- ncol(Y.sel)
  set.seed(123)
  train.index <- apply(Y.sel, 2, function(x) (train_test_split_j(x)))
  observed <- matrix(1, n, p)
  test_mask <- matrix(0, n, p)
  val_mask <- matrix(0, n, p)
  
  Y.train <- Y.sel
  for(j in 1:p){
    test <- train.index[[j]]$test
    val <- train.index[[j]]$val
    not_train <- c(test)
    if(length(val)>0){not_train <- c(not_train, val)}
    Y.train[not_train, j] <- NA
    if(sum(Y.sel[-not_train, j]==1)==0){
      print('error')
    }
    if(sum(Y.sel[not_train, j]==1)==0){
      print('error')
    }
    observed[not_train, j] <- 0
    test_mask[test, j] <- 1
    val_mask[val, j] <- 1
  }
  train.set <- which(!is.na(Y.train))
  test.val.set <- which(is.na(Y.train))
  test.set <- which(test_mask == 1)
  val.set <- which(val_mask == 1)
  
  Y.sel.train.imputed.c <- Y.train %>% 
    as.data.frame() %>% 
    mutate_all(~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x))  %>%
    as.matrix()
  Y.sel.train.imputed.r <- t(Y.train) %>% 
    as.data.frame() %>% 
    mutate_all(~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x))  %>%
    as.matrix() %>% t()
  
  Y.sel.train.imputed <- as.matrix(Y.train)
  Y.sel.train.imputed[test.val.set] <- Y.sel.train.imputed.r[test.val.set]*Y.sel.train.imputed.c[test.val.set]
  rm(Y.sel.train.imputed.r, Y.sel.train.imputed.c)
  return(list(
    observed = observed,
    test_mask = test_mask,
    val_mask = val_mask,
    train_set = train.set,
    test_set = test.set,
    val_set = val.set,
    Y_train_imputed = Y.sel.train.imputed,
    Y_sel = Y.sel,
    Y_train = Y.train
  ))
}


compute_auc <- function(Y, X, Beta, M, Lambda, test_set, val_set, train_set){
  
  Z_hat <- X %*% t(as.matrix(Beta)) + M %*% t(Lambda)
  
  p_hat_test <- 1/(1+exp(-Z_hat[test_set]))
  roc_test <- roc(c(Y[test_set]), c(p_hat_test), direction="<", quiet=TRUE)
  auc_test <- auc(roc_test)[1]
  rm(roc_test)
  
  p_hat_val <- 1/(1+exp(-Z_hat[val_set]))
  roc_val <- roc(c(Y[val_set]), c(p_hat_val), direction="<", quiet=TRUE)
  auc_val <- auc(roc_val)[1]
  rm(roc_val)
  
  test_val_set <- c(test_set, val_set)
  p_hat_test_val <- 1/(1+exp(-Z_hat[test_val_set]))
  roc_test_val <- roc(c(Y[test_val_set]), c(p_hat_test_val), direction="<", quiet=TRUE)
  auc_test_val <- auc(roc_test_val)[1]
  rm(roc_test_val)
  
  p_hat_train <- 1/(1+exp(-Z_hat[train_set]))
  roc_train <- roc(c(Y[train_set]), c(p_hat_train), direction="<", quiet=TRUE)
  auc_train <- auc(roc_train)[1]
  rm(roc_train)
  
  print('AUC')
  print(paste('test =', auc_test, 'val =', auc_val, 'hold-out =', auc_test_val, 'train =', auc_train))
  
  return(c(auc_test, auc_val, auc_test_val, auc_train))
}

# Z.hat.insects.1 <- X %*% t(as.matrix(flair.insects.th15$Beta_bar)) + flair.insects.th15$M_tilde %*% t(flair.insects.th15$Lambda_bar)
# p.hat.insects.test.1 <- 1/(1+exp(-Z.hat.insects.1[test.set.th15]))
# roc.insects.test.1 <- roc(c(Y.sel.th15[test.set.th15]), c(p.hat.insects.test.1))
# auc.insects.test.1 <- auc(roc.insects.test.1)[1]; auc.insects.test.1
# rm(roc.insects.test.1)
# 
# p.hat.insects.val.1 <- 1/(1+exp(-Z.hat.insects.1[val.set.th15]))
# roc.insects.val.1 <- roc(c(Y.sel.th15[val.set.th15]), c(p.hat.insects.val.1))
# auc.insects.val.1 <- auc(roc.insects.val.1)[1]; auc.insects.val.1
# rm(roc.insects.val.1)
# 
# p.hat.insects.holdout.1 <- 1/(1+exp(-Z.hat.insects.1[test.val.set.th15]))
# roc.insects.holdout.1 <- roc(c(Y.sel.th15[test.val.set.th15]), c(p.hat.insects.holdout.1))
# auc.insects.holdout.1 <- auc(roc.insects.holdout.1)[1]; auc.insects.holdout.1
# rm(roc.insects.holdout.1)
