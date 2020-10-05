setwd('/home/marcos/Documentos/01_IAA/IAA007_008_MachineLearning/Praticas/04_Biomassa_Regressao/')
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
dados <- read.csv("Biomassa_Dados.csv")

dados
summary(dados)

# tratando valores missing
#dados$id <- NULL
#imp <- mice(dados) # pacote mice trata dados ausentes, inputa os dados ausentes "m" vezes
#dados <- complete(imp,1) # exporta dados imputados
#dados

# Cria arquivo de treino e teste
set.seed(37)
indices <- createDataPartition(dados$biomassa, p=0.80, list=FALSE) 
treino <- dados[indices,]
teste <- dados[-indices,]

############################### RNA - Regressão ##########################

# Treinar o modelo com Hold-Out
set.seed(37)
rna <- train(biomassa~., data=treino, method="nnet", linout=T, trace=FALSE)
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

rmse(teste$biomassa, predicoes.rna)
# 278.7978

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
  }

r2(teste$biomassa, predicoes.rna)
"
> r2(teste$biomassa, predicoes.rna)
R2 com hold-out
0.9193191
"

# Cross Validation e parametrização da RNA
control <- trainControl(method = "cv", number = 10)
tuneGrid <- expand.grid(size = seq(from = 1, to = 10, by = 1),
                        decay = seq(from = 0.1, to = 0.9, by = 0.3))

set.seed(37)
rna <- train(biomassa~., data=treino, method="nnet", trainControl=control,
             tuneGrid=tuneGrid, linout=T, MaxNWts=10000, maxit=2000, trace=F)
rna
"
>rna
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were size = 7 and decay = 0.4.
"

# Predições e métricas aplicadas na base teste
predicoes.rna <- predict(rna, teste)
rmse(teste$biomassa, predicoes.rna)
"
> rmse(teste$biomassa, predicoes.rna)
309.9738
"

r2(teste$biomassa, predicoes.rna)
"
R2 com Cross Validation parametrizado
> r2(teste$biomassa, predicoes.rna)
0.9000082
"

# Predições de novos casos
dados_novos_casos <- read.csv("Biomassa_Dados_Novos_Casos.csv")
dados_novos_casos

"
> dados_novos_casos
dap   h    Me    biomassa
6.4   7.0  1.04  ?
7.3   10.0 1.04  ?
7.8   5.5  1.04  ?
12.2  7.5  1.04  ?
"

dados_novos_casos$biomassa <- NULL
predict.rna <- predict(rna, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predict.rna)
resultado

"
> resultado
dap   h    Me    predict.rna
6.4   7.0  1.04  27.73007
7.3   10.0 1.04  52.60684
7.8   5.5  1.04  27.08128
12.2  7.5  1.04  66.26544
"

############################### KNN - Regressão #########################

# Prepara um grid com os valores de k que serão usados 
tuneGrid <- expand.grid(k = c(1,3,5,7,9))

# Executa o Knn com o grid criado
set.seed(37)
knn <- train(biomassa ~ ., data = treino, method = "knn", tuneGrid=tuneGrid)
knn

# Aplica o modelo no arquivo de teste
predict.knn <- predict(knn, teste)

# Carrega a biblioteca e calcula as Métricas
library(Metrics)
rmse(teste$biomassa, predict.knn)

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$biomassa, predict.knn)
"
> r2(teste$biomassa, predict.knn)
[1] 0.6059138
"

# Predições de novos casos
dados_novos_casos <- read.csv("Biomassa_Dados_Novos_Casos.csv")
dados_novos_casos

predict.knn <- predict(knn, dados_novos_casos)
dados_novos_casos$biomassa <- NULL
resultado <- cbind(dados_novos_casos, predict.knn)
resultado
"
> resultado
   dap    h   Me predict.knn
1  6.4  7.0 1.04       12.79
2  7.3 10.0 1.04        7.84
3  7.8  5.5 1.04       13.90
4 12.2  7.5 1.04       46.75
"

############################## SVM - Regressão ##########################
# Treinar SVM com a base de Treino
set.seed(37)
svm <- train(biomassa~., data=treino, method="svmRadial") 
svm

"
>svm
Tuning parameter 'sigma' was held constant at a value of 0.7220883
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.7220883 and C = 1.
"
# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$biomassa, predicoes.svm)
"
> rmse(teste$biomassa, predicoes.svm)
[1] 406.4227
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$biomassa, predicoes.svm)
"
> r2(teste$biomassa, predicoes.svm)
[1] 0.8281087
"

# Cross-validation SVM
ctrl <- trainControl(method = "cv", number = 10)

set.seed(37)
svm <- train(biomassa~., data=treino, method="svmRadial", trControl=ctrl)
svm
"
Tuning parameter 'sigma' was held constant at a value of 0.7220883
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.7220883 and C = 1.
"

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$biomassa, predicoes.svm)
"
> rmse(teste$biomassa, predicoes.svm)
[1] 406.4227
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$biomassa, predicoes.svm)
"
> r2(teste$biomassa, predicoes.svm)
[1] 0.8281087
"

# Vários C e sigma
tuneGrid = expand.grid(C=c(1, 2, 10, 50, 100), sigma=c(.01, .015, 0.2))

set.seed(37)
svm <- train(biomassa~., data=treino, method="svmRadial", trControl=ctrl, tuneGrid=tuneGrid)
svm
"
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.01 and C = 100.
"

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$biomassa, predicoes.svm)
"
> rmse(teste$biomassa, predicoes.svm)
[1] 308.6657
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$biomassa, predicoes.svm)
"
> r2(teste$biomassa, predicoes.svm)
[1] 0.9018723
"

# Predições de novos casos
dados_novos_casos <- read.csv("Biomassa_Dados_Novos_Casos.csv")
dados_novos_casos

dados_novos_casos$biomassa <- NULL
predict.svm <- predict(svm, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predict.svm)
resultado
"
> resultado
   dap    h   Me predict.svm
1  6.4  7.0 1.04    159.2163
2  7.3 10.0 1.04    150.5539
3  7.8  5.5 1.04    165.1988
4 12.2  7.5 1.04    157.8682
"

