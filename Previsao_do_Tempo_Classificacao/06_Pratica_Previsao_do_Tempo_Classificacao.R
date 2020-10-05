setwd('/home/marcos/Documentos/01_IAA/IAA007_008_MachineLearning/Praticas/06_Pratica_Previsao_do_Tempo/')
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
dados <- read.csv('Material 02 - 6 - C - Previsao do Tempo - Dados.csv')

# analisando o dataframe
dados
summary(dados)
names(dados)

# tratando valores missing
#dados$a <- NULL # excluir a coluna Índice
#imp <- mice(dados)
#dados <- complete(imp,1)

# separando base treino - 80% e teste 20%
set.seed(37)
indices <- createDataPartition(dados$Chovera, p=0.80, list=FALSE)
treino <- dados[indices,]
teste <- dados[-indices,]

# treinamento do modelo com o conjunto de treino
set.seed(37)
rna <-  train(Chovera~.,data=treino, method="nnet",trace=FALSE)
rna

# predições dos valores do conjunto de teste
predict.rna <- predict(rna,teste)

# matriz de confusão
confusionMatrix(predict.rna, teste$Chovera)

# Usando Cross-Validation
ctrl <- trainControl(method = "cv", number = 10)

# executa a RNA com esse ctrl
set.seed(37)
rna <- train(Chovera~.,
             data = treino,
             method = "nnet", 
             trace = FALSE,
             trControl=ctrl)

# parametrização RNA
# size, decay
grid <- expand.grid(size = seq(from = 1, to = 45, by=10),
                    decay = seq(from=0.1, to=0.9, by=0.3))

set.seed(37)
rna <- train(
  form = Chovera~.,
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
confusionMatrix(predict.rna, teste$Chovera)

