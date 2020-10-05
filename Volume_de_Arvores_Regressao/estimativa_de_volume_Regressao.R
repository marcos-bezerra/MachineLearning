setwd('/home/marcos/Documentos/01_IAA/IAA007_008_MachineLearning/Praticas/03_Estimativa_de_Volume_Regressao/')
getwd()

# instalar bibliotecas
install.packages("caret")
install.packages("e1071")
install.packages("mlbench")
install.packages("mice")

library(mlbench)
library(caret)
library(mice)

# leitura dos dados da base de volumes
dados <- read.csv("Estimativa_de_Volume_Dados.csv", header = T)

dados
summary(dados)
names(dados)

# Cria arquivo de treino e teste
set.seed(1912)
indices <- createDataPartition(dados$Volume, p=0.80, list=FALSE) 
treino <- dados[indices,]
teste <- dados[-indices,]

############################### RNA - Regressão ##########################

# Treino com Hold-Out
set.seed(1912)
rna <- train(Volume~., data=treino, method="nnet", linout=T, trace=FALSE)
rna
predicoes.rna <- predict(rna, teste)

# Pacote para cálculo das métricas (rmse)
#install.packages("Metrics")
library(Metrics)

rmse(teste$Volume, predicoes.rna)

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
  }

r2(teste$Volume, predicoes.rna) # R2 com hold-out

# CV e parametrização da RNA
control <- trainControl(method = "cv", number = 10)
tuneGrid <- expand.grid(size = seq(from = 1, to = 10, by = 1), decay = seq(from = 0.1, to = 0.9, by = 0.3))

set.seed(1912)
rna <- train(Volume~., data=treino, method="nnet", trainControl=control, tuneGrid=tuneGrid, linout=T, MaxNWts=10000, maxit=2000, trace=F)
rna

# Predições e métricas na base teste
predicoes.rna <- predict(rna, teste)
rmse(teste$Volume, predicoes.rna)
r2(teste$Volume, predicoes.rna) # R2 com Cross Validation parametrizado

# Predições de novos casos
dados_novos_casos <- read.csv("Estimativa_de_Volume_Dados_Novos_Casos.csv")
dados_novos_casos

dados_novos_casos$Volume <- NULL
predict.rna <- predict(rna, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predict.rna)
resultado


############################### KNN - Regressão ###########################

### Prepara um grid com os valores de k que serão usados 
tuneGrid <- expand.grid(k = c(1,3,5,7,9))

### Executa o Knn com esse grid
set.seed(1912)
knn <- train(Volume ~ ., data = treino, method = "knn", tuneGrid=tuneGrid)
knn

# Aplica o modelo no arquivo de teste
predict.knn <- predict(knn, teste)

# Mostra as métricas
library(Metrics)
rmse(teste$Volume, predict.knn)

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$Volume, predict.knn)

# Predições de novos casos
dados_novos_casos <- read.csv("Material 02 - 3 – Estimativa de Volume - Dados - Novos Casos.csv")
dados_novos_casos

predict.knn <- predict(knn, dados_novos_casos)
dados_novos_casos$Volume <- NULL
resultado <- cbind(dados_novos_casos, predict.knn)
resultado


############################## SVM - Regressão ##########################
# Treinar SVM com a base de Treino 
set.seed(1912)
svm <- train(Volume~., data=treino, method="svmRadial") 
svm

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$Volume, predicoes.svm)


r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$Volume, predicoes.svm)

# Cross-validation SVM
ctrl <- trainControl(method = "cv", number = 10)

set.seed(1912)
svm <- train(Volume~., data=treino, method="svmRadial", trControl=ctrl)
svm

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$Volume, predicoes.svm)


r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$Volume, predicoes.svm)

# Vários C e sigma
tuneGrid = expand.grid(C=c(1, 2, 10, 50, 100), sigma=c(.01, .015, 0.2))

set.seed(1912)
svm <- train(Volume~., data=treino, method="svmRadial", trControl=ctrl, tuneGrid=tuneGrid)
svm

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$Volume, predicoes.svm)


r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$Volume, predicoes.svm)

# Predições de novos casos
dados_novos_casos <- read.csv("Estimativa_de_Volume_Dados_Novos_Casos.csv")
dados_novos_casos

dados_novos_casos$Volume <- NULL
predict.svm <- predict(svm, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predict.svm)
resultado

