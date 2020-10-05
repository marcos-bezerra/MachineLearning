setwd('/home/marcos/Documentos/01_IAA/IAA007_008_MachineLearning/Praticas/07_Pratica_Imposto_de_Renda/')
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
dados <- read.csv('Material 02 - 7 – C - IR - Dados.csv')

# analisando o dataframe
dados
summary(dados)
names(dados)

# tratando valores missing
#dados$Id <- NULL # excluir a coluna "Id"
#imp <- mice(dados)
#dados <- complete(imp,1)

# separando base treino - 80% e teste 20%
set.seed(37)
indices <- createDataPartition(dados$sonegador, p=0.80, list=FALSE)
treino <- dados[indices,]
teste <- dados[-indices,]

# treinamento do modelo com o conjunto de treino
set.seed(37)
rna <-  train(sonegador~.,data=treino, method="nnet",trace=FALSE)
rna

# predições dos valores do conjunto de teste
predict.rna <- predict(rna,teste)

# matriz de confusão
confusionMatrix(predict.rna, teste$sonegador)

# Usando Cross-Validation
ctrl <- trainControl(method = "cv", number = 10)

# executa a RNA com esse ctrl
set.seed(37)
rna <- train(sonegador~., data = treino, method = "nnet", trace = FALSE,trControl=ctrl)

# parametrização RNA
# size, decay
grid <- expand.grid(size = seq(from = 1, to = 45, by=10),
                    decay = seq(from=0.1, to=0.9, by=0.3))

set.seed(37)
rna <- train(
  form = sonegador~.,
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
confusionMatrix(predict.rna, teste$sonegador)

#########################################################################
# predições de novos casos
dados_novos_casos <- read.csv("Material 02 - 7 – C - IR - Dados - Novos Casos.csv")
dados_novos_casos
names(dados_novos_casos)

predict.rna <- predict(rna, dados_novos_casos)

dados_novos_casos$sonegador <- NULL
resultado <- cbind(dados_novos_casos, predict.rna)
resultado

# executar um modelo com os melhores hiperparâmetros
grid <- expand.grid(size = c(1), decay = c(0.1))

set.seed(37)
melhor_modelo_rna <- train(form = sonegador~.,
                           data = dados,
                           method = 'nnet',
                           tuneGrid = grid,
                           trControl= ctrl,
                           maxit = 2000,
                           trace=FALSE)

# Salvar melhor modelo
saveRDS (melhor_modelo_rna, "Material 02 - 7 – C - IR - Dados - Novos Casos.rds")

# ler e aplicar o modelo
modelo_lido <- readRDS('./Material 02 - 7 – C - IR - Dados - Novos Casos.rds')
novas_predicoes <- predict(modelo_lido, teste)
confusionMatrix(novas_predicoes, as.factor(teste$sonegador))
