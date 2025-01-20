install.packages("mlbench")
install.packages("ggplot")

library(caret) #ML Model buidling package
library(tidyverse) #ggplot and dplyr
library(MASS) #Modern Applied Statistics with S
library(mlbench) #data sets from the UCI repository.
library(summarytools)
library(corrplot) #Correlation plot
library(gridExtra) #Multiple plot in single grip space
library(timeDate) 
library(pROC) #ROC
library(caTools) #AUC
library(rpart.plot) #CART Decision Tree
library(e1071) #imports graphics, grDevices, class, stats, methods, utils
library(graphics) #fourfoldplot

df <- read.csv("C:/Users/flori/Downloads/diabetes.csv")
df$diabetes <- factor(df$diabetes, levels = c("neg", "pos"))
str(df)


#store rows for partition
partition <- caret::createDataPartition(y = df$diabetes, times = 1, p = 0.7, list = FALSE)

# create training data set
train_set <- df[partition,]

# create testing data set, subtracting the rows partition to get remaining 30% of the data
test_set <- df[-partition,]

str(train_set)

str(test_set)

summary(train_set)

summarytools::descr(train_set)


ggplot(train_set, aes(train_set$diabetes, fill = diabetes)) + 
  geom_bar() +
  theme_bw() +
  labs(title = "Diabetes Classification", x = "Diabetes") +
  theme(plot.title = element_text(hjust = 0.5))

cor_data <- cor(train_set[,setdiff(names(train_set), 'diabetes')])
#Numerical Correlation Matrix
cor_data


# Correlation matrix plots
corrplot::corrplot(cor_data)

corrplot::corrplot(cor_data, type = "lower", method = "number")


corrplot::corrplot(cor_data, type = "lower", method = "pie")


univar_graph <- function(univar_name, univar, data, output_var) {
  
  g_1 <- ggplot(data, aes(x=univar)) + 
    geom_density() + 
    xlab(univar_name) + 
    theme_bw()
  
  g_2 <- ggplot(data, aes(x=univar, fill=output_var)) + 
    geom_density(alpha=0.4) + 
    xlab(univar_name) + 
    theme_bw()
  
  gridExtra::grid.arrange(g_1, g_2, ncol=2, top = paste(univar_name,"variable", "/ [ Skew:",timeDate::skewness(univar),"]"))
  
}

for (x in 1:(ncol(train_set)-1)) {
  univar_graph(univar_name = names(train_set)[x], univar = train_set[,x], data = train_set, output_var = train_set[,'diabetes'])
}


# ggplot(data = train_set, aes(x = age, fill = diabetes)) +
#   geom_bar(stat='count', position='dodge') +
#   ggtitle("Age Vs Diabetes") +
#   theme_bw() +
#   labs(x = "Age") +
#   theme(plot.title = element_text(hjust = 0.5))  

bivar_graph <- function(bivar_name, bivar, data, output_var) {
  
  g_1 <- ggplot(data = data, aes(x = bivar, fill = output_var)) +
    geom_bar(stat='count', position='dodge') +
    theme_bw() +
    labs( title = paste(bivar_name,"- Diabetes", sep =" "), x = bivar_name) +
    theme(plot.title = element_text(hjust = 0.5))
  
  plot(g_1)
}

for (x in 1:(ncol(train_set)-1)) {
  bivar_graph(bivar_name = names(train_set)[x], bivar = train_set[,x], data = train_set, output_var = train_set[,'diabetes'])
}

#outlier detection
box_plot <- function(bivar_name, bivar, data, output_var) {
  
  g_1 <- ggplot(data = data, aes(y = bivar, fill = output_var)) +
    geom_boxplot() +
    theme_bw() +
    labs( title = paste(bivar_name,"Outlier Detection", sep =" "), y = bivar_name) +
    theme(plot.title = element_text(hjust = 0.5))
  
  plot(g_1)
}

for (x in 1:(ncol(train_set)-1)) {
  box_plot(bivar_name = names(train_set)[x], bivar = train_set[,x], data = train_set, output_var = train_set[,'diabetes'])
}

# Random Forest Model
model_forest <- caret::train(diabetes ~., data = train_set,
                             method = "ranger",
                             metric = "ROC",
                             trControl = trainControl(method = "cv", number = 10,
                                                      classProbs = T, summaryFunction = twoClassSummary),
                             preProcess = c("center","scale","pca"))
model_forest

# final ROC Value
model_forest$results[6,4]


xgb_grid_1  <-  expand.grid(
  nrounds = 50,
  eta = c(0.03),
  max_depth = 1,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 1,
  subsample = 0.5
)

model_xgb <- caret::train(diabetes ~., data = train_set,
                          method = "xgbTree",
                          metric = "ROC",
                          tuneGrid=xgb_grid_1,
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))
model_xgb

#KNN 
model_knn <- caret::train(diabetes ~., data = train_set,
                         method = "knn",
                         metric = "ROC",
                         tuneGrid = expand.grid(.k = c(3:10)),
                         trControl = trainControl(method = "cv", number = 10,
                                                  classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))

model_knn

# Logistic Regression
model_glm <- caret::train(diabetes ~., data = train_set,
                          method = "glm",
                          metric = "ROC",
                          tuneLength = 10,
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))

model_glm

#final ROC Value
model_glm$results[2]


model_list <- list(Random_Forest = model_forest, XGBoost = model_xgb, KNN = model_knn, Logistic_Regression = model_glm)
resamples <- resamples(model_list)

#box plot
bwplot(resamples, metric="ROC")

#dot plot
dotplot(resamples, metric="ROC")

# Random Forest
# prediction on Test data set
pred_rf <- predict(model_forest, test_set)
# Confusion Matrix 
cm_rf <- confusionMatrix(pred_rf, test_set$diabetes, positive="pos")


# Prediction Probabilities
pred_prob_rf <- predict(model_forest, test_set, type="prob")
# ROC value
roc_rf <- roc(test_set$diabetes, pred_prob_rf$pos)


# Confusion Matrix for Random Forest Model
cm_rf

# ROC Value for Random Forest
roc_rf

# AUC - Area under the curve
caTools::colAUC(pred_prob_rf$pos, test_set$diabetes, plotROC = T)

# XGBOOST - eXtreme Gradient BOOSTing
# prediction on Test data set
pred_xgb <- predict(model_xgb, test_set)
# Confusion Matrix 
cm_xgb <- confusionMatrix(pred_xgb, test_set$diabetes, positive="pos")
# Prediction Probabilities
pred_prob_xgb <- predict(model_xgb, test_set, type="prob")
# ROC value
roc_xgb <- roc(test_set$diabetes, pred_prob_xgb$pos)
# Confusion matrix 
cm_xgb

# ROC Value for for XGBoost
roc_xgb

# AUC - Area under the curve
caTools::colAUC(pred_prob_xgb$pos, test_set$diabetes, plotROC = T)

# KNN - K Nearest Neighbours
# prediction on Test data set
pred_knn <- predict(model_knn, test_set)
# Confusion Matrix 
cm_knn <- confusionMatrix(pred_knn, test_set$diabetes, positive="pos")
# Prediction Probabilities
pred_prob_knn <- predict(model_knn, test_set, type="prob")
# ROC value
roc_knn <- roc(test_set$diabetes, pred_prob_knn$pos)
# Confusion matrix 
cm_knn

# ROC Value for for XGBoost
roc_knn

# AUC - Area under the curve
caTools::colAUC(pred_prob_knn$pos, test_set$diabetes, plotROC = T)

# Logistic Regression
# prediction on Test data set
pred_glm <- predict(model_glm, test_set)
# Confusion Matrix 
cm_glm <- confusionMatrix(pred_glm, test_set$diabetes, positive="pos")
# Prediction Probabilities
pred_prob_glm <- predict(model_glm, test_set, type="prob")
# ROC value
roc_glm <- roc(test_set$diabetes, pred_prob_glm$pos)
# Confusion matrix 
cm_glm

# ROC Value for for XGBoost
roc_glm

# AUC - Area under the curve
caTools::colAUC(pred_prob_glm$pos, test_set$diabetes, plotROC = T)

#Results
result_rf <- c(cm_rf$byClass['Sensitivity'], cm_rf$byClass['Specificity'], cm_rf$byClass['Precision'], 
               cm_rf$byClass['Recall'], cm_rf$byClass['F1'], roc_rf$auc)

result_xgb <- c(cm_xgb$byClass['Sensitivity'], cm_xgb$byClass['Specificity'], cm_xgb$byClass['Precision'], 
                cm_xgb$byClass['Recall'], cm_xgb$byClass['F1'], roc_xgb$auc)

result_knn <- c(cm_knn$byClass['Sensitivity'], cm_knn$byClass['Specificity'], cm_knn$byClass['Precision'], 
                cm_knn$byClass['Recall'], cm_knn$byClass['F1'], roc_knn$auc)

result_glm <- c(cm_glm$byClass['Sensitivity'], cm_glm$byClass['Specificity'], cm_glm$byClass['Precision'], 
                cm_glm$byClass['Recall'], cm_glm$byClass['F1'], roc_glm$auc)

all_results <- data.frame(rbind(result_rf, result_xgb, result_knn, result_glm))
names(all_results) <- c("Sensitivity", "Specificity", "Precision", "Recall", "F1", "AUC")
all_results



col <- c("#ed3b3b", "#0099ff")

graphics::fourfoldplot(cm_rf$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("Random Forest Accuracy(",round(cm_rf$overall[1]*100),"%)", sep = ""))

graphics::fourfoldplot(cm_xgb$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("XGB Accuracy(",round(cm_xgb$overall[1]*100),"%)", sep = ""))

graphics::fourfoldplot(cm_knn$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("KNN Accuracy(",round(cm_knn$overall[1]*100),"%)", sep = ""))

graphics::fourfoldplot(cm_glm$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("Logistic Regression Accuracy(",round(cm_glm$overall[1]*100),"%)", sep = ""))