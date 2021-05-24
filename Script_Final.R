library(readr)
library(mlbench)
library(plyr)
library(caret)
library(ggplot2)
library(smotefamily)
library(randomForest)
library(rpart)
library(dplyr)
library(FactoMineR)
library(ggcorrplot)
library(rpart.plot)

confusion_matrix <- function(y_test, model, x_test){
  
  conf.matrix <- table(y_test, 
                       predict(model, newdata = x_test, type="class"))
  
  rownames(conf.matrix) <- paste("Actual", 
                                 rownames(conf.matrix), sep = ":")
  
  colnames(conf.matrix) <- paste("Pred", 
                                 colnames(conf.matrix), sep = ":")
  
  return(conf.matrix)
}

class_scoring <- function(conf_matrix){
  
  recall <- conf_matrix[1,1]/(conf_matrix[1,1] + 
                                conf_matrix[1,2])
  
  precision <- conf_matrix[1,1]/(conf_matrix[1,1] +
                                   conf_matrix[2,1])
  
  accuracy <- (conf_matrix[1,1]+conf_matrix[2,2])/
    (conf_matrix[1,1]+conf_matrix[1,2] + 
       conf_matrix[2,1]+conf_matrix[2,2])
  
  f1_score <- 2*((precision*recall)
                 /(precision+recall))
  
  results <- c(recall, accuracy, precision, f1_score)
  
  df <- data.frame(results)
  
  rownames(df) <- c('Recall', 'Accuracy', 'Precision', 'F1 Score')
  
  return(df)
  
}


BankChurners <- read_csv("BankChurners.csv")

BankChurners$Attrition_Flag <- revalue(BankChurners$Attrition_Flag, 
                                       c("Existing Customer"= "Active", 
                                         "Attrited Customer"="Inactive"))

data <- BankChurners

data[, c(1,3,5:8)] <- lapply(data[, c(1,3,5:8)], 
                             as.factor)

summary(data)

# some graphs about categorical variables wrt the target

ggplot(data) +
  aes(x = Education_Level, fill = Attrition_Flag) +
  geom_bar() +
  scale_fill_hue() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 0.5, hjust=0.5))

ggplot(data) +
  aes(x = Income_Category, fill = Attrition_Flag) +
  geom_bar() +
  scale_fill_hue() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 0.5, hjust=0.5))

ggplot(data) +
  aes(x = Card_Category, fill = Attrition_Flag) +
  geom_bar() +
  scale_fill_hue() +
  theme_minimal()

ggplot(data) +
  aes(x = Marital_Status, fill = Attrition_Flag) +
  geom_bar() +
  scale_fill_hue() +
  theme_minimal()

ggplot(data) +
  aes(x = Gender, fill = Attrition_Flag) +
  geom_bar() +
  scale_fill_hue() +
  theme_minimal()

seed = 4321
set.seed(seed)

cor_spearman <- cor(data[, sapply(data, is.numeric)], 
                    method = 'spearman')

as.matrix(data.frame(cor_spearman)) %>% 
  round(3) %>% #round
  ggcorrplot('square','full', lab = TRUE)

# Split data into training (80%) and test (20%)
dt = sort(sample(nrow(BankChurners), nrow(BankChurners)*.8))
train<-BankChurners[dt,]
test<-BankChurners[-dt,]

train[, c(1,3,5:8)] <- lapply(train[, c(1,3,5:8)], as.factor)
test[, c(1,3,5:8)] <- lapply(test[, c(1,3,5:8)], as.factor)

### FIRST TREE ###


tree.mod = rpart(formula = Attrition_Flag ~ ., data = train)

plotcp(tree.mod)

tree.mod = prune.rpart(tree.mod, 0.023)

rpart.plot(tree.mod, fallen.leaves = FALSE, 
           extra = 109, under = TRUE, 
           tweak = 1.6, clip.facs = TRUE, 
           compress = FALSE, space = 0.35)

a <- caret::confusionMatrix(data = predict(tree.mod, 
                                           newdata = test, type="class"), 
                            reference = test$Attrition_Flag, 
                            mode = 'everything')

matrix1 <- confusion_matrix(test$Attrition_Flag, 
                            model = tree.mod, 
                            x_test = test)

scores1 <- class_scoring(matrix1)

### FIRST RANDOM FOREST ###

rf <- randomForest(Attrition_Flag~., 
                   data=train, importance=TRUE)

varImpPlot(rf, main = "Important Predictors in Random Forest")

b <- caret::confusionMatrix(data = predict(rf, 
                                           newdata = test, type="class"),
                            reference = test$Attrition_Flag, 
                            mode = 'everything')

matrix2 <- confusion_matrix(test$Attrition_Flag, 
                            model = rf, 
                            x_test = test)
scores2 <- class_scoring(matrix2)

#################################################
###              model selection              ###
#################################################

train.top = subset(train, select=c(Attrition_Flag,Total_Trans_Ct,
                                   Total_Trans_Amt,Total_Relationship_Count,
                                   Total_Ct_Chng_Q4_Q1,Total_Amt_Chng_Q4_Q1,
                                   Total_Revolving_Bal))

test.top = subset(test, select=c(Attrition_Flag,Total_Trans_Ct,
                                 Total_Trans_Amt,Total_Relationship_Count, 
                                 Total_Ct_Chng_Q4_Q1,Total_Amt_Chng_Q4_Q1,
                                 Total_Revolving_Bal))


### SECOND TREE ###

tree.mod.top = rpart(formula = Attrition_Flag ~., 
                     data = train.top)

plotcp(tree.mod.top)

tree.mod.top = prune(tree.mod.top, cp = 0.011)

rpart.plot(tree.mod.top, fallen.leaves = FALSE, 
           extra = 109, under = TRUE, 
           tweak = 1.7, clip.facs = TRUE, 
           compress = TRUE, space = 0.35)

c <- caret::confusionMatrix(data = predict(tree.mod.top, 
                                           newdata = test.top, type="class"), 
                            reference = test.top$Attrition_Flag, 
                            mode = 'everything')

matrix3 <- confusion_matrix(test$Attrition_Flag, 
                            model = tree.mod.top, 
                            x_test = test.top)

scores3 <- class_scoring(matrix3)

### SECOND RANDOM FOREST ###

rf.top <- randomForest(formula, data=train.top, 
                       importance=FALSE)

d <- caret::confusionMatrix(data = predict(rf.top, 
                                           newdata = test.top, type="class"), 
                            reference = test.top$Attrition_Flag, 
                            mode = 'everything')

matrix4 <- confusion_matrix(test$Attrition_Flag, 
                            model = tree.mod.top, 
                            x_test = test.top)

scores4 <- class_scoring(matrix4)

#################################################
###                  Tuning                   ###
#################################################

### TUNED RANDOM FOREST WITH MODEL SELECTION ###

tuned.rf1 <- tune.randomForest(x = train.top[, -1], 
                               y = train.top[, c('Attrition_Flag')], 
                               ntree = c(500, 1000, 1500, 2000), 
                               mtry = c(sqrt(ncol(train.top)), 4, 6))

e <- caret::confusionMatrix(data = 
                              predict(tuned.rf1$best.model, 
                                      newdata = test.top, type="class"), 
                            reference = test.top$Attrition_Flag, 
                            mode = 'everything')

matrix4 <- confusion_matrix(test.top$Attrition_Flag, 
                            model = tuned.rf1$best.model, 
                            x_test = test.top)

scores4 <- class_scoring(matrix4)

### TUNED RANDOM FOREST WITHOUT MODEL SELECTION ###

tuned.rf1bis <- tune.randomForest(x = train[, -1], 
                                  y = train$Attrition_Flag, 
                                  ntree = c(500, 1000, 1500, 2000), 
                                  mtry = c(sqrt(ncol(train)), 5, 10, 15, 20))

f <- caret::confusionMatrix(data = 
                              predict(tuned.rf1bis$best.model, 
                                      newdata = test.top, type="class"), 
                            reference = test.top$Attrition_Flag, 
                            mode = 'everything')

matrix5 <- confusion_matrix(test.top$Attrition_Flag, 
                            model = tuned.rf1bis$best.model, 
                            x_test = test.top)

scores5 <- class_scoring(matrix5)


##############################################
###      SMOTE on the training test        ###
##############################################

train.smote <- SMOTE(target = train.top$Attrition_Flag, 
                     X = as.data.frame(train.top[,-1]),
                     K = 3, dup_size = 2)
train.smote <- rbind(train.smote$data, 
                     train.smote$syn_data)
names(train.smote)[names(train.smote) == 'class'] <- 'Attrition_Flag'
train.smote[, 7] <- as.factor(train.smote[, 7])

prob.or <- round((prop.table(table(train$Attrition_Flag))*100), 2)
orig <- data.frame(prob.or)


pie = ggplot(orig, aes(x="", y=Freq, fill=Var1)) + 
  geom_bar(stat="identity", width=1)

pie = pie + coord_polar("y", start=0) + 
  geom_text(aes(label = paste0(Freq, "%")), 
            position = position_stack(vjust = 0.5))

pie = pie + labs(x = NULL, y = NULL, 
                 fill = NULL, 
                 title = "original target data")

pie = pie + theme_classic() + 
  theme(axis.line = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        plot.title = element_text(hjust = 0.5))

# pie chart with balanced classes

prob.smote <- round((prop.table(table(train.smote$Attrition_Flag))*100), 2)
smote <- data.frame(prob.smote)

pie1 = ggplot(smote, aes(x="", y=Freq, fill=Var1)) + 
  geom_bar(stat="identity", width=1)

pie1 = pie1 + 
  coord_polar("y", start=0) + 
  geom_text(aes(label = paste0(Freq, "%")), 
            position = position_stack(vjust = 0.5))

pie1 = pie1 + 
  labs(x = NULL, y = NULL, 
       fill = NULL, title = "balanced target data")

pie1 = pie1 + 
  theme_classic() + 
  theme(axis.line = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        plot.title = element_text(hjust = 0.5))

### THIRD DECISION TREE with SMOTE ###

tree.mod2 = rpart(formula = Attrition_Flag ~ ., data = train.smote)

plotcp(tree.mod2)
tree.mod2 = prune(tree.mod2, cp=0.016)


g <- caret::confusionMatrix(data = predict(tree.mod2, 
                                           newdata = test.top, type="class"), 
                            reference = test.top$Attrition_Flag, 
                            mode = 'everything')

matrix6 <- confusion_matrix(test.top$Attrition_Flag, 
                            model = tree.mod2, 
                            x_test = test.top)

scores6 <- class_scoring(matrix6)

### TUNED SMOOTE RANDOM FOREST ###

tuned.rf2 <- tune.randomForest(x = train.smote[, -7], 
                               y = train.smote[, c('Attrition_Flag')], 
                               ntree = c(500, 1000, 1500, 2000), 
                               mtry = c(sqrt(ncol(train.smote)), 4, 6))

tuned.rf2$best.model

h <- caret::confusionMatrix(data = predict(tuned.rf2$best.model, 
                                           newdata = test.top, type="class"), 
                            reference = test.top$Attrition_Flag, mode = 'everything')

matrix7 <- confusion_matrix(test.top$Attrition_Flag, 
                            model = tuned.rf2, 
                            x_test = test.top)

scores7 <- class_scoring(matrix7)