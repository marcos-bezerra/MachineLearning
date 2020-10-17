setwd('/home/marcos/Documentos/01_IAA/IAA007_008_MachineLearning/Praticas/MachineLearning/Cancer_de_Mama_Classificacao/')
getwd()

# importando bibliotecas
#library(e1071)
install.packages("mice")
library(mice)

#install.packages("caret")
#install.packages("e1071")
library(caret)

#install.packages("mlbench")
library(mlbench)

# Importar conjunto de dados
dados <- read.csv('Cancer_de_Mama_Dados.csv')

# analisando o dataframe
summary(dados)
View(dados)
names(dados)

# tratando valores missing
dados$Id <- NULL # excluir a coluna "Id"
#imp <- mice(dados)
#dados <- complete(imp,1)

# separando base treino - 80% e teste 20%
set.seed(1912)
indices <- createDataPartition(dados$Class, p=0.80, list=FALSE)
#indices <- sample(1:nrow(dados),0.8 * nrow(dados))
treino <- dados[indices,]
teste <- dados[-indices,]

############################ RNA -  Classificação #######################

# treinamento do modelo com o conjunto de treino
set.seed(1912)
rna <-  train(Class~.,data=treino, method="nnet",trace=FALSE)
rna

# predições dos valores do conjunto de teste
predict.rna <- predict(rna,teste)

# matriz de confusão
confusionMatrix(predict.rna, teste$Class)

# Usando Cross-Validation
ctrl <- trainControl(method = "cv", number = 10)

# executa a RNA com esse ctrl
set.seed(1912)
rna <- train(Class~., data = treino, method = "nnet", trace = FALSE,trControl=ctrl)

# parametrização RNA
# size, decay
grid <- expand.grid(size = seq(from = 1, to = 45, by=10), decay = seq(from=0.1, to=0.9, by=0.3))

set.seed(1912)
rna <- train(
  form = Class~.,
  data = treino,
  method = 'nnet',
  tuneGrid = grid,
  trControl = ctrl,
  maxit = 2000,
  trace = FALSE
  )

# verificar o resultado do treinamento
rna

# predições dos valores do conjunto de teste
predict.rna <- predict(rna,teste)

# matriz de confusão
confusionMatrix(predict.rna, teste$Class)

# predições de novos casos
dados_novos_casos <- read.csv("Cancer_de_Mama_Dados_Novos_Casos.csv")
names(dados_novos_casos)
#dados_novos_casos$Id <- NULL
dados_novos_casos

predict.rna <- predict(rna, dados_novos_casos)
predict.rna

dados_novos_casos$Class <- NULL
dados_novos_casos$Id <- NULL
resultado <- cbind(dados_novos_casos, predict.rna)
resultado

confusionMatrix(predict.rna,resultado$predict.rna)

# executar um modelo com os melhores hiperparâmetros
grid <- expand.grid(size = c(1), decay = c(0.1))

set.seed(1912)
melhor_modelo_rna <- train(form = Class~.,
                           data = dados,
                           method = 'nnet',
                           tuneGrid = grid,
                           trControl= ctrl,
                           maxit = 2000,
                           trace=FALSE)

# Salvar melhor modelo
saveRDS (melhor_modelo_rna, "Cancer_de_Mama_R_Melhor_Modelo_RNA.rds")

# ler e aplicar o modelo
modelo_lido <- readRDS('./Cancer_de_Mama_R_Melhor_Modelo_RNA.rds')
novas_predicoes <- predict(modelo_lido, teste)
confusionMatrix(novas_predicoes, as.factor(teste$Class))


########################### KNN -  Classificação ########################

### Faz um grid com valores para K e Executa o KNN
tuneGrid <- expand.grid(k = c(1,3,5,7,9))

set.seed(1912)
knn <- train(Class ~ ., data = treino, method = "knn",tuneGrid=tuneGrid)
knn

# Faz a predição e mostra a matriz de confusão
predict.knn <- predict(knn, teste)
confusionMatrix(predict.knn, as.factor(teste$Class))

# PREDIÇÕES DE NOVOS CASOS
dados_novos_casos <- read.csv("Cancer_de_Mama_Dados_Novos_Casos.csv")
dados_novos_casos$Id <- NULL
dados_novos_casos

predict.knn <- predict(knn, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predict.knn)
dados_novos_casos$Class <- NULL
resultado

# EXECUTAR UM MODELO COM OS MELHORES HIPERPARÂMETROS
tuneGrid <- expand.grid(k = c(9))

set.seed(1912)
melhor_modelo_knn <- train(Class ~ ., data = dados, method = "knn",tuneGrid=tuneGrid)
melhor_modelo_knn

# SALVAR O MELHOR MODELO PARA USO NA PRÁTICA

#SALVAR O MODELO
getwd()
saveRDS(melhor_modelo_knn,"Cancer_de_Mama_R_Melhor_Modelo_KNN.rds")

# LER E APLICAR O MODELO
modelo_lido <- readRDS("./Cancer_de_Mama_R_Melhor_Modelo_KNN.rds")
novas_predicoes <- predict(modelo_lido, teste)
confusionMatrix(novas_predicoes, as.factor(teste$Class))

############################ SVM -  Classificação ########################

# Treinar SVM com a base de Treino 
set.seed(1912)
svm <- train(Class~., data=treino, method="svmRadial") 
svm

# Aplicar modelos treinados na base de Teste
predict.svm <- predict(svm, teste)
confusionMatrix(predict.svm, as.factor(teste$Class))

# Cross-validation SVM
ctrl <- trainControl(method = "cv", number = 10)

set.seed(1912)
svm <- train(Class~., data=treino, method="svmRadial", trControl=ctrl)
svm

# matriz de confusao com todos os dados
predict.svm <- predict(svm, teste)
confusionMatrix(predict.svm, as.factor(teste$Class))

# Vários C e sigma
tuneGrid = expand.grid(C=c(1, 2, 10, 50, 100), sigma=c(.01, .015, 0.2))

set.seed(1912)
svm <- train(Class~., data=treino, method="svmRadial", trControl=ctrl, tuneGrid=tuneGrid)
svm

# matriz de confusao com todos os dados
predict.svm <- predict(svm, teste)
confusionMatrix(predict.svm, as.factor(teste$Class))

# Predições de novos casos
dados_novos_casos <- read.csv("Cancer_de_Mama_Dados_Novos_Casos.csv") 
dados_novos_casos$Id <- NULL
dados_novos_casos

predict.svm <- predict(svm, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predict.svm)
resultado$Class <- NULL
resultado

# EXECUTAR UM MODELO COM OS MELHORES HIPERPARÂMETROS
tuneGrid = expand.grid(C=c(10), sigma=c(.015))

set.seed(1912)
melhor_modelo_svm <- train(Class~., data=dados, method="svmRadial", trControl=ctrl, tuneGrid=tuneGrid)
melhor_modelo_svm

# SALVAR O MELHOR MODELO PARA USO NA PRÁTICA
getwd()
saveRDS(melhor_modelo_svm,"Cancer_de_Mama_R_Melhor_Modelo_SVM.rds")

# LER E APLICAR O MODELO
modelo_lido <- readRDS("./Cancer_de_Mama_R_Melhor_Modelo_SVM.rds")
novas_predicoes <- predict(modelo_lido, teste)
confusionMatrix(novas_predicoes, as.factor(teste$Class))
