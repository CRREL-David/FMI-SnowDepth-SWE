#Developed by David Brodylo from USACE CRREL-AKRO. For questions contact david.brodylo@usace.army.mil.
#Purpose is to aid in estimating snow depth and snow water equivalent (SWE) with CNN, RF, SVM, ANN, LM, and an ensemble (EA) model.
#Made with R version 4.5.0

#Load necessary libraries
library(keras3)
tensorflow::set_random_seed(42)
library(caret)
library(Metrics)
library(randomForest)
library(kernlab)

setwd("C:/Users/...") #Set the working directory.
samples <- read.csv(".csv") #Specify .csv file with image objects that contain matched field and remote sensing data.
Data <- read.csv(".csv") #Specify .csv file with all image objects that have data.

set.seed(42)
#CNN
x_0 <- as.matrix(samples[,-91]) #Predictors.
y_0 <- as.matrix(samples[,91])  #Target variable.
x_rows <- as.matrix(nrow(x_0))  #Number of rows.

x_Data <- as.matrix(Data[,-91])  #Predictors for Data.

pca_result <- prcomp(x_0, center = TRUE, scale. = TRUE) #Principal Component Analysis (PCA).
x <- pca_result$x

#Center and scale the new data using the means and standard deviations from the training data.
preProc_Data <- scale(x_Data, center = pca_result$center, scale = pca_result$scale)
#Compute the principal component scores for the new data.
x_FinalData <- as.matrix(preProc_Data) %*% pca_result$rotation

#Define k-fold cross-validation.
k <- 10
folds <- sample(rep(1:k, length.out = x_rows))

#Initialize a vector to store results.
cv_results <- numeric(k)

#Function to loop through each fold.
for (i in 1:k) {
  cat("Processing fold", i, "\n")
  
  #Split data into training and validation sets.
  train_indices <- which(folds != i)
  val_indices <- which(folds == i)
  
  x_train <- x[train_indices, ]
  y_train <- y_0[train_indices]
  x_val <- x[val_indices, ]
  y_val <- y_0[val_indices]
  
  #Generate the 1D CNN model.
  model <- keras_model_sequential() %>%
    layer_conv_1d(filters = 32, kernel_size = 2, activation = "relu", input_shape = c(ncol(x), 1)) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear") # Regression output
  
  #Compile the model.
  model %>% compile(
    optimizer = "adam",
    loss = "mse",  # Mean Squared Error for regression
    metrics = list("r2_score")  # R2 score
  )
  
  #Monitor val_loss and stop training when no improvement is evident with patience at 20.
  early_stopping <- callback_early_stopping(
    monitor = "val_loss",
    min_delta = 0,
    patience = 20,
    mode='auto',
    restore_best_weights = TRUE
  )
  
  #Train the model.
  history <- model %>% fit(
    x_train, y_train,
    epochs = 100,
    batch_size = 4,
    validation_data = list(x_val, y_val),
    callbacks = list(early_stopping),
    verbose = 1
  )
  
  #Evaluate the model on the validation set.
  results <- model %>% evaluate(x_val, y_val, verbose = 0)
  cv_results[i] <- results$r2_score
}

#Predict the final target variable from the sample data after all folds were run. 
ypred = model %>% predict(x)

#Compare predicted outputs with field data.
x_axes = seq(1:length(ypred))
plot(x_axes, y_0, ylim = c(min(ypred), max(y_0)),
     col = "burlywood", type = "l", lwd = 2, xlab = "Predicted Snow Depth", ylab = "Field Snow Depth")
lines(x_axes, ypred, col = "red", type = "l", lwd = 2)
legend("topleft", legend = c("y-test", "y-pred"),
       col = c("burlywood", "red"), lty = 1, cex=0.7, lwd=2, bty='n')

cnn_res1 <- data.frame(Observed = y_0) #Field data.
cnn_res2 <- data.frame(Predicted = ypred) #Predicted data.
cnn_res3 <- cbind(cnn_res1, cnn_res2) #Field and predicted data.
cnn_LR <- lm(Observed ~ Predicted, data = cnn_res3) #Generate linear relationship between the input and estimated outputs.

plot(x = cnn_res3$Predicted, y = cnn_res3$Observed) #Plot the points and linear relationship.
abline (a=0, b=1)
abline(cnn_LR, col = "red1", lwd = 2)

cnn_name = "cnn" #Model name.
cnn_rSquared = format(signif(summary(lm(Observed ~ Predicted, data = cnn_res3))$r.squared, digits = 2), nsmall = 2) #Model Rsquared.
cnn_MAE = format(signif(mae(cnn_res3$Observed, cnn_res3$Predicted), digits = 2), nsmall = 1)  #Model MAE.
cnn_RMSE = format(signif(rmse(cnn_res3$Observed, cnn_res3$Predicted), digits = 2), nsmall = 1)  #Model RMSE.
Model_Stats <- data.frame(Model = cnn_name, rSquared = cnn_rSquared, MAE = cnn_MAE, RMSE = cnn_RMSE) #Combined model metrics.

#Machine Learning Models.
#Define k-fold cross-validation.
control <- trainControl(method = "cv",
                        number = 10,
                        savePredictions = "final",
                        allowParallel = TRUE)

set.seed(42)
#RF
rf_Model <- train(Class  ~ .,
                  data = samples,
                  method = "rf",
                  preProcess = c("center", "scale", "pca"), #Principal Component Analysis (PCA).
                  trControl = control,
                  metric = "Rsquared",
                  ntree = 20, #Number of trees
                  tuneGrid = expand.grid(mtry = c(1, 2, 3, 4))) #Number of variable splits.

rf_res1 <- data.frame(Observed = samples$Class) #Field data
rf_res2 <- data.frame(Predicted = predict(rf_Model)) #Predicted data.
rf_res3 <- cbind(rf_res1, rf_res2) #Field and predicted data.
rf_LR <- lm(Observed ~ Predicted, data = rf_res3) #Generate linear relationship between the input and estimated outputs.

plot(x = rf_res3$Predicted, y = rf_res3$Observed) #Plot the points and linear relationship.
abline (a=0, b=1)
abline(rf_LR, col = "red1", lwd = 2)

rf_name = "rf" #Model name.
rf_rSquared = format(signif(summary(lm(Observed ~ Predicted, data = rf_res3))$r.squared, digits = 2), nsmall = 2) #Model Rsquared.
rf_MAE = format(signif(mae(rf_res3$Observed, rf_res3$Predicted), digits = 2), nsmall = 1) #Model MAE.
rf_RMSE = format(signif(rmse(rf_res3$Observed, rf_res3$Predicted), digits = 2), nsmall = 1) #Model RMSE.
Model_Stats <- rbind(Model_Stats, data.frame(Model = rf_name, rSquared = rf_rSquared, MAE = rf_MAE, RMSE = rf_RMSE)) #Combined model metrics.

set.seed(42)
#SVM
svm_Model <- train(Class  ~ .,
                   data = samples,
                   method = "svmPoly",
                   preProcess = c("center", "scale", "pca"), #Principal Component Analysis (PCA).
                   trControl = control,
                   metric = "Rsquared",
                   tuneGrid = expand.grid(degree = c(2, 3), #Polynomial degrees.
                                          C = c(0.01, 0.1, 1.0, 10.0, 100.0), #Cost parameter.
                                          scale = c(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)), #Scaling parameter.
                   decision.values = TRUE)

svm_res1 <- data.frame(Observed = samples$Class) #Field data.
svm_res2 <- data.frame(Predicted = predict(svm_Model)) #Predicted data.
svm_res3 <- cbind(svm_res1, svm_res2) #Field and predicted data.

svm_LR <- lm(Observed ~ Predicted, data = svm_res3) #Generate linear relationship between the input and estimated outputs.

plot(x = svm_res3$Predicted, y = svm_res3$Observed) #Plot the points and linear relationship.
abline (a=0, b=1)
abline(svm_LR, col = "red1", lwd = 2)

svm_name = "svm" #Model name.
svm_rSquared = format(signif(summary(lm(Observed ~ Predicted, data = svm_res3))$r.squared, digits = 2), nsmall = 2) #Model Rsquared.
svm_MAE = format(signif(mae(svm_res3$Observed, svm_res3$Predicted), digits = 2), nsmall = 1) #Model MAE.
svm_RMSE = format(signif(rmse(svm_res3$Observed, svm_res3$Predicted), digits = 2), nsmall = 1) #Model RMSE.
Model_Stats <- rbind(Model_Stats, data.frame(Model = svm_name, rSquared = svm_rSquared, MAE = svm_MAE, RMSE = svm_RMSE)) #Combined model metrics.

set.seed(42)
#ANN
nnet_Model <- train(Class  ~ .,
                    data = samples,
                    method = "nnet",
                    preProcess = c("center", "scale", "pca"), #Principal Component Analysis (PCA).
                    trControl = control,
                    metric = "Rsquared",
                    tuneGrid = expand.grid(size = seq(from = 2, to = 8, by = 1), #Size of units in hidden layer.
                                           decay = seq(from = 0.01, to = 0.1, by = 0.01)), #Decay parameter.
                    maxit = 200, #Maximum iterations.
                    abstol = 1e-4, #Stop if the fit criterion goes below abstol.
                    linout = TRUE)

nnet_res1 <- data.frame(Observed = samples$Class) #Field data.
nnet_res2 <- data.frame(Predicted = predict(nnet_Model)) #Predicted data.
nnet_res3 <- cbind(nnet_res1, nnet_res2) #Field and predicted data.
nnet_LR <- lm(Observed ~ Predicted, data = nnet_res3) #Generate linear relationship between the input and estimated outputs.

plot(x = nnet_res3$Predicted, y = nnet_res3$Observed) #Plot the points and linear relationship.
abline (a=0, b=1)
abline(nnet_LR, col = "red1", lwd = 2)

nnet_name = "nnet" #Model name.
nnet_rSquared = format(signif(summary(lm(Observed ~ Predicted, data = nnet_res3))$r.squared, digits = 2), nsmall = 2) #Model Rsquared.
nnet_MAE = format(signif(mae(nnet_res3$Observed, nnet_res3$Predicted), digits = 2), nsmall = 1) #Model MAE.
nnet_RMSE = format(signif(rmse(nnet_res3$Observed, nnet_res3$Predicted), digits = 2), nsmall = 1) #Model RMSE.
Model_Stats <- rbind(Model_Stats, data.frame(Model = nnet_name, rSquared = nnet_rSquared, MAE = nnet_MAE, RMSE = nnet_RMSE)) #Combined model metrics.

set.seed(42)
#LM
lm_Model <- train(Class  ~ .,
                  data = samples,
                  method = "lm",
                  preProcess = c("center", "scale", "pca"), #Principal Component Analysis (PCA).
                  trControl = control,
                  metric = "Rsquared")

lm_res1 <- data.frame(Observed = samples$Class) #Field data.
lm_res2 <- data.frame(Predicted = predict(lm_Model)) #Predicted data.
lm_res3 <- cbind(lm_res1, lm_res2) #Field and predicted data.

lm_LR <- lm(Observed ~ Predicted, data = lm_res3) #Generate linear relationship between the input and estimated outputs.

plot(x = lm_res3$Predicted, y = lm_res3$Observed) #Plot the points and linear relationship.
abline (a=0, b=1)
abline(lm_LR, col = "red1", lwd = 2)

lm_name = "lm" #Model name.
lm_rSquared = format(signif(summary(lm(Observed ~ Predicted, data = lm_res3))$r.squared, digits = 2), nsmall = 2) #Model Rsquared.
lm_MAE = format(signif(mae(lm_res3$Observed, lm_res3$Predicted), digits = 2), nsmall = 1) #Model MAE.
lm_RMSE = format(signif(rmse(lm_res3$Observed, lm_res3$Predicted), digits = 2), nsmall = 1) #Model RMSE.
Model_Stats <- rbind(Model_Stats, data.frame(Model = lm_name, rSquared = lm_rSquared, MAE = lm_MAE, RMSE = lm_RMSE)) #Combined model metrics.
R2_Num <- as.numeric(Model_Stats$rSquared) #R2 values.

set.seed(42)
## EA ##
#Combined model outputs from RF, SVM, and ANN are used to determine the ensemble (EA) outcome based on their R2 value.
Model_Weights <- R2_Num[1] + R2_Num[2] + R2_Num[3] + R2_Num[4] + R2_Num[5] #Generate the weight from each of the base models based off of the respective R2 value of each.
CNN_weight = (R2_Num[1]/Model_Weights) #Weight for CNN
RF_weight = (R2_Num[2]/Model_Weights) #Weight for RF
SVM_weight = (R2_Num[3]/Model_Weights) #Weight for SVM
NNET_weight = (R2_Num[4]/Model_Weights) #Weight for NNET
LM_weight = (R2_Num[5]/Model_Weights) #Weight for LM

ea_res1 <- ((cnn_res2 * CNN_weight) + (predict(rf_Model) * RF_weight) + (predict(svm_Model) * SVM_weight) + (predict(nnet_Model) * NNET_weight)+ (predict(lm_Model) * LM_weight)) #Generates the final EA outputs.
ea_res2 <- data.frame(cbind(Observed = samples$Class, Predicted = ea_res1)) #Creates data frame of the sample input values plus the EA outputs.
ea_LR <- lm(Observed ~ Predicted, data = ea_res2) #Generate linear relationship between the input and estimated outputs.

plot(x = ea_res2$Predicted, y = ea_res2$Observed) #Plot the points and linear relationship.
abline (a=0, b=1)
abline(ea_LR, col = "red1", lwd = 2)

ea_name = "ea" #Model name.
ea_rSquared = format(signif(summary(lm(Observed ~ Predicted, data = ea_res2))$r.squared, digits = 2), nsmall = 2) #Model Rsquared.
ea_MAE = format(signif(mae(ea_res2$Observed, ea_res2$Predicted), digits = 2), nsmall = 1) #Model MAE.
ea_RMSE = format(signif(rmse(ea_res2$Observed, ea_res2$Predicted), digits = 2), nsmall = 1) #Model RMSE.
Model_Stats <- rbind(Model_Stats, data.frame(Model = ea_name, rSquared = ea_rSquared, MAE = ea_MAE, RMSE = ea_RMSE)) #Combined model metrics.

Model_Results_STD <- data.frame(cnn = cnn_res3$Predicted, rf = rf_res3$Predicted, svm = svm_res3$Predicted, nnet = nnet_res3$Predicted, lm = lm_res3$Predicted) #Standard deviation values from models.
Model_STD <- transform(Model_Results_STD, STD = apply(Model_Results_STD, 1, sd, na.rm = TRUE))

Model_Results <- cbind(Model_STD, data.frame(ea = ea_res2$Predicted, Observed = samples$Class)) #Combined model metrics
Model_Stats_transpose <- as.data.frame(t(Model_Stats))

set.seed(42)
## PREDICITONS ##
#Generates the estimated values for all image objects in the study area.
Pred_1 <- data.frame(data.frame(model %>% predict(x_FinalData)), #Runs the CNN model on the image objects.
                     data.frame(predict(rf_Model, Data)), #Runs the RF model on the image objects.
                     data.frame(predict(svm_Model, Data)), #Runs the SVM model on the image objects.
                     data.frame(predict(nnet_Model, Data)), #Runs the ANN model on the image objects.
                     data.frame(predict(lm_Model, Data))) #Runs the LM model on the image objects.
Pred_1[Pred_1 < 0] <- 0 #Remove any negative predicted values.

#Find the highest value in the 'sample' column.
max_value <- max(samples$Class)
#Define the threshold as 1.5x the highest value.
threshold <- 1.5 * max_value
Pred_1[Pred_1 > threshold] <- threshold #Replace any unusually high predicted values.
Pred_2 <- transform(Pred_1, STD = apply(Pred_1, 1, sd, na.rm = TRUE))

Pred_3 <- cbind(Pred_2, (Pred_1$model.....predict.x_FinalData. * CNN_weight) + (Pred_1$predict.rf_Model..Data. * RF_weight) + (Pred_1$predict.svm_Model..Data. * SVM_weight) + (Pred_1$predict.nnet_Model..Data. * NNET_weight) + (Pred_1$predict.lm_Model..Data. * LM_weight)) #Generates the final EA outputs.
names(Pred_3) <- c("CNN", #Updates the header names
                   "RF",
                   "SVM",
                   "NNET",
                   "MLR",
                   "STD",
                   "EA")
write.csv(Pred_3, ".csv", row.names = FALSE) #Export final predicted results from all models.
write.csv(Model_Results, ".csv", row.names = FALSE) #Export final model metrics from all models.
