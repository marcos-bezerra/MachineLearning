setwd('/home/marcos/Documentos/01_IAA/IAA007_008_MachineLearning/Praticas/MachineLearning/00_Iris_Classificacao/Iris/')
getwd()

# importando bibliotecas
library(e1071)
library(caret)

# Importar conjunto de dados
dados <- (iris)

# analisando o dataframe
summary(dados)
View(dados)

# separando base treino - 80% e teste 20%
set.seed(1912)
indices <- createDataPartition(dados$Species, p=0.80, list=FALSE)
#indices <- sample(1:nrow(dados), 0.8* nrow(dados))
treino <- dados[indices,]
teste <- dados[-indices,]

########################### RNA - Classificação ##########################

# treinamento do modelo com o conjunto de treino
set.seed(1912)
rna <-  train(Species~.,data=treino, method="nnet",linout=T,trace=FALSE)
rna

# predições dos valores do conjunto de teste
predict.rna <- predict(rna,teste)

# matriz de confusão
confusionMatrix(predict.rna, teste$Species)

# Salvar conjunto de dados no formato R
save (rna, file = "rna.RData")

# Carregar conjunto de dados salvo no formato do R
load("rna.RData")

########################### KNN - Classificação ##########################

# cria um grid com vários valores para K e faz o treinamento
tuneGrid <- expand.grid(k = c(1,3,5,7,9))

set.seed(1912)
knn <-  train(Species ~., data = treino, method = "knn", tuneGrid = tuneGrid)
knn

# Faz a predição e mostra a matriz de confusão
predict.knn <- predict(knn, teste)
confusionMatrix(predict.knn, as.factor(teste$Species))

# Salvar conjunto de dados no formato R
save (rna, file = "knn.RData")

# Carregar conjunto de dados salvo no formato do R
load("knn.RData")

########################### SVM - Classificação ##########################

# Gerar um novo modelo usando SVM, predições e matriz de confusão
set.seed(1912)
svm <- train(Species~., data=treino, method="svmRadial") 
svm

# Predições com o arquivo de teste
predicoes.svm <- predict(svm, teste)
confusionMatrix(predicoes.svm, teste$Species)

# Salvar conjunto de dados no formato R
save (rna, file = "svm.RData")

# Carregar conjunto de dados salvo no formato do R
load("svm.RData")

###########################  - Classificação ##########################