setwd('/home/marcos/Documentos/01_IAA/IAA007_008_MachineLearning/Praticas/MachineLearning/Admissao_Regressao/')
getwd()

# instalar bibliotecas
install.packages("caret")
install.packages("e1071")
install.packages("mlbench")
install.packages("mice")

# carregando as bibliotecas
library(mlbench)
library(caret)
library(mice)

# carregando a base de dados
dados <- read.csv("https://raw.githubusercontent.com/marcos-bezerra/MachineLearning/main/Admissao_Regressao/Admissao_Dados.csv")

View(dados)
summary(dados)

# tratando valores missing
dados$Serial.No. <- NULL
imp <- mice(dados) # pacote mice trata dados ausentes, inputa os dados ausentes "m" vezes
dados <- complete(imp,1) # exporta dados imputados
View(dados)

# Cria arquivo de treino e teste
set.seed(37)
indices <- createDataPartition(dados$ChanceOfAdmit, p=0.80, list=FALSE) 
treino <- dados[indices,]
teste <- dados[-indices,]

############################### RNA - Regressão ##########################

# Treinar o modelo com Hold-Out
set.seed(37)
rna <- train(ChanceOfAdmit~., data=treino, method="nnet", linout=T, trace=FALSE)
rna
"
> rna
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were size = 5 and decay = 0.1.
"

# Calculo das predições para a base teste
predicoes.rna <- predict(rna, teste)

# Instalando pacote para cálculo das métricas Metrics - (rmse)
install.packages("Metrics")
library(Metrics)

rmse(teste$ChanceOfAdmit, predicoes.rna)
"
> rmse(teste$ChanceOfAdmit, predicoes.rna)
[1] 0.07313229
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}

r2(teste$ChanceOfAdmit, predicoes.rna)
"
> r2(teste$ChanceOfAdmit, predicoes.rna)
R2 com hold-out
[1] 0.7404921
"

# Cross Validation e parametrização da RNA
control <- trainControl(method = "cv", number = 10)
tuneGrid <- expand.grid(size = seq(from = 1, to = 10, by = 1),
                        decay = seq(from = 0.1, to = 0.9, by = 0.3))

set.seed(37)
rna <- train(ChanceOfAdmit~., data=treino, method="nnet", trainControl=control,
             tuneGrid=tuneGrid, linout=T, MaxNWts=10000, maxit=2000, trace=F)
rna
"
>rna
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were size = 10 and decay = 0.1.
"

# Predições e métricas aplicadas na base teste
predicoes.rna <- predict(rna, teste)
rmse(teste$ChanceOfAdmit, predicoes.rna)
"
> rmse(teste$biomassa, predicoes.rna)
[1] 0.06824529
"

r2(teste$ChanceOfAdmit, predicoes.rna)
"
R2 com Cross Validation parametrizado
> r2(teste$ChanceOfAdmit, predicoes.rna)
0.7740104
"

# Predições de novos casos
dados_novos_casos <- read.csv("Admissao_Dados_Novos_Casos.csv")
dados_novos_casos

dados_novos_casos$Serial.No. <- NULL
predict.rna <- predict(rna, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predict.rna)
resultado


############################### KNN - Regressão #########################

# Prepara um grid com os valores de k que serão usados 
tuneGrid <- expand.grid(k = c(1,3,5,7,9))

# Executa o Knn com o grid criado
set.seed(37)
knn <- train(ChanceOfAdmit ~ ., data = treino, method = "knn", tuneGrid=tuneGrid)
knn

"
RMSE was used to select the optimal model using the smallest value.
The final value used for the model was k = 9.
"

# Aplica o modelo no arquivo de teste
predict.knn <- predict(knn, teste)

# Carrega a biblioteca e calcula as Métricas
library(Metrics)
rmse(teste$ChanceOfAdmit, predict.knn)

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$ChanceOfAdmit, predict.knn)
"
> r2(teste$ChanceOfAdmit, predict.knn)
[1] 0.6588849
"

# Predições de novos casos
dados_novos_casos <- read.csv("Admissao_Dados_Novos_Casos.csv")
View(dados_novos_casos)

dados_novos_casos$Serial.No. <- NULL
predict.knn <- predict(knn, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predict.knn)
resultado


############################## SVM - Regressão ##########################
# Treinar SVM com a base de Treino
set.seed(37)
svm <- train(ChanceOfAdmit~., data=treino, method="svmRadial") 
svm

"
>svm
Tuning parameter 'sigma' was held constant at a value of 0.1894597
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.1894597 and C = 0.5.
"
# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$ChanceOfAdmit, predicoes.svm)
"
> rmse(teste$ChanceOfAdmit, predicoes.svm)
[1] 0.07006415
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$ChanceOfAdmit, predicoes.svm)
"
> r2(teste$ChanceOfAdmit, predicoes.svm)
[1] 0.7620843
"

# Cross-validation SVM
ctrl <- trainControl(method = "cv", number = 10)

set.seed(37)
svm <- train(ChanceOfAdmit~., data=treino, method="svmRadial", trControl=ctrl)
svm
"
Tuning parameter 'sigma' was held constant at a value of 0.1894597
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.1894597 and C = 1.
"

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$ChanceOfAdmit, predicoes.svm)
"
> rmse(teste$ChanceOfAdmit, predicoes.svm)
[1] 0.06953861
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$ChanceOfAdmit, predicoes.svm)
"
> r2(teste$ChanceOfAdmit, predicoes.svm)
[1] 0.7655416
"

# Vários C e sigma
tuneGrid = expand.grid(C=c(1, 2, 10, 50, 100), sigma=c(.01, .015, 0.2))

set.seed(37)
svm <- train(ChanceOfAdmit~., data=treino, method="svmRadial", trControl=ctrl, tuneGrid=tuneGrid)
svm
"
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.01 and C = 50.
"

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$ChanceOfAdmit, predicoes.svm)
"
> rmse(teste$ChanceOfAdmit, predicoes.svm)
[1] 0.06729262
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$ChanceOfAdmit, predicoes.svm)
"
> r2(teste$ChanceOfAdmit, predicoes.svm)
[1] 0.7805504
"

# Predições de novos casos
dados_novos_casos <- read.csv("Admissao_Dados_Novos_Casos.csv")
dados_novos_casos

dados_novos_casos$Serial.No. <- NULL
predict.svm <- predict(svm, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predict.svm)
resultado

