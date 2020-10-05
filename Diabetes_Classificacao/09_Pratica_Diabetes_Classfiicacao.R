setwd('/home/marcos/Documentos/01_IAA/IAA007_008_MachineLearning/Praticas/09_Pratica_Diabetes/')
getwd()

### CLASSIFICAÇÃO

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
dados <- read.csv('09_Pratica_Diabetes.csv')

# analisando o dataframe
summary(dados)
dados
names(dados)

# tratando valores missing
dados$num <- NULL # excluir a coluna "Id"
imp <- mice(dados)
dados <- complete(imp,1)

dados$pregnt[which(dados$pregnt==0)] <- "NA"
dados$glucose[dados$glucose == 0] <- NA
dados[,1:5][dados[,1:5]==0] <- "NA"
dados

# separando base treino - 80% e teste 20%
set.seed(37)
indices <- createDataPartition(dados$diabetes, p=0.80, list=FALSE)
treino <- dados[indices,]
teste <- dados[-indices,]

# treinamento do modelo com o conjunto de treino
set.seed(37)
rna <-  train(diabetes~.,data=treino, method="nnet",trace=FALSE)
rna

# predições dos valores do conjunto de teste
predict.rna <- predict(rna,teste)

# matriz de confusão
confusionMatrix(predict.rna, teste$diabetes)

# Usando Cross-Validation
ctrl <- trainControl(method = "cv", number = 10)

# executa a RNA com esse ctrl
set.seed(37)
rna <- train(diabetes~., data = treino, method = "nnet", trace = FALSE,trControl=ctrl)

# parametrização RNA
# size, decay
grid <- expand.grid(size = seq(from = 1, to = 45, by=10),
                    decay = seq(from=0.1, to=0.9, by=0.3))

set.seed(37)
rna <- train(
  form = diabetes~.,
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
confusionMatrix(predict.rna, teste$diabetes)

