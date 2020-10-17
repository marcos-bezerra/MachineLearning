setwd('/home/marcos/Documentos/01_IAA/IAA007_008_MachineLearning/Praticas/MachineLearning/Biomassa_Regressao/')
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
dados <- read.csv("https://raw.githubusercontent.com/marcos-bezerra/MachineLearning/main/Biomassa_Regressao/biomassa_dados.csv")

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
Neural Network 

240 samples
  3 predictor

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 240, 240, 240, 240, 240, 240, ... 
Resampling results across tuning parameters:

  size  decay  RMSE       Rsquared   MAE     
  1     0e+00  1359.6765  0.2254846  486.4328
  1     1e-04  1355.6432  0.5218986  490.8272
  1     1e-01   972.5392  0.6590893  323.4447
  3     0e+00  1267.4186  0.4347920  460.8710
  3     1e-04  1316.6751  0.2501682  486.5336
  3     1e-01  1150.3640  0.5575517  343.4845
  5     0e+00  1326.0819  0.3288325  461.2318
  5     1e-04  1416.7994  0.1261932  495.0866
  5     1e-01   903.3091  0.6080807  277.8309

RMSE was used to select the optimal model using the smallest value.
The final values used for the model were size = 5 and decay = 0.1
"

# Calculo das predições para a base teste
predicoes.rna <- predict(rna, teste)
predicoes.rna

# Instalando pacote para cálculo das métricas Metrics - (rmse)
#install.packages("Metrics")
library(Metrics)

rmse(teste$biomassa, predicoes.rna)
"
[1] 278.7978
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
  }

r2(teste$biomassa, predicoes.rna)
"
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
Neural Network 

240 samples
  3 predictor

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 240, 240, 240, 240, 240, 240, ... 
Resampling results across tuning parameters:

  size  decay  RMSE      Rsquared   MAE     
   1    0.1    992.6191  0.6905533  349.0330
   1    0.4    908.6370  0.8317616  280.6225
   1    0.7    941.4060  0.7374847  300.7257
   2    0.1    782.2559  0.7571487  233.3292
   2    0.4    737.4780  0.8414254  186.4817
   2    0.7    777.7248  0.7304216  232.9334
   3    0.1    849.6777  0.7436672  222.6957
   3    0.4    795.0072  0.7796057  192.7300
   3    0.7    695.3849  0.8570516  162.9742
   4    0.1    849.6070  0.7662399  220.9410
   4    0.4    720.7256  0.8588244  165.3920
   4    0.7    698.7138  0.8685893  156.4966
   5    0.1    797.7404  0.8294112  199.5125
   5    0.4    683.2916  0.8567750  164.9856
   5    0.7    718.4193  0.8460354  170.1174
   6    0.1    776.6870  0.7621348  203.8486
   6    0.4    716.9926  0.8330087  171.5059
   6    0.7    689.5052  0.8577947  157.8576
   7    0.1    773.4558  0.8268441  194.1351
   7    0.4    662.8995  0.8665862  154.5710
   7    0.7    685.0345  0.8598895  165.3012
   8    0.1    773.0733  0.7758522  190.2286
   8    0.4    695.6610  0.8524127  164.6003
   8    0.7    693.9999  0.8421501  173.3723
   9    0.1    767.2943  0.8154317  197.6349
   9    0.4    737.0334  0.8473800  174.8264
   9    0.7    738.1415  0.8299597  180.3076
  10    0.1    766.9103  0.8256608  181.6695
  10    0.4    746.3551  0.8556866  173.5491
  10    0.7    687.5034  0.8663602  161.5511

RMSE was used to select the optimal model using the smallest value.
The final values used for the model were size = 7 and decay = 0.4.
"

# Predições e métricas aplicadas na base teste
predicoes.rna <- predict(rna, teste)
rmse(teste$biomassa, predicoes.rna)
"
[1] 309.9738
"

r2(teste$biomassa, predicoes.rna)
"
R2 com Cross Validation parametrizado
[1] 0.9000082
"

"
Análise final:
R2 com Hold-out = 0.9193191
R2 com CV parametrizado = 0.9000082
"

# Predições de novos casos
dados_novos_casos <- read.csv("https://raw.githubusercontent.com/marcos-bezerra/MachineLearning/main/Biomassa_Regressao/biomassa_dados_novos_casos.csv")
dados_novos_casos
"
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
"
k-Nearest Neighbors 

240 samples
  3 predictor

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 240, 240, 240, 240, 240, 240, ... 
Resampling results across tuning parameters:

  k  RMSE      Rsquared   MAE     
  1  612.5019  0.9069416  137.0779
  3  762.8607  0.8535735  155.7368
  5  859.1033  0.8094264  163.6333
  7  897.7325  0.7906541  168.4094
  9  939.1502  0.7590357  173.6134

RMSE was used to select the optimal model using the smallest value.
The final value used for the model was k = 1.
"

# Aplica o modelo no arquivo de teste
predict.knn <- predict(knn, teste)

# Carrega a biblioteca e calcula as Métricas
library(Metrics)
rmse(teste$biomassa, predict.knn)
"
[1] 616.0497
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$biomassa, predict.knn)
"
[1] 0.6059138
"

# Predições de novos casos
dados_novos_casos <- read.csv("https://raw.githubusercontent.com/marcos-bezerra/MachineLearning/main/Biomassa_Regressao/biomassa_dados_novos_casos.csv")
dados_novos_casos

predict.knn <- predict(knn, dados_novos_casos)
dados_novos_casos$biomassa <- NULL
resultado <- cbind(dados_novos_casos, predict.knn)
resultado
"
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
Support Vector Machines with Radial Basis Function Kernel 

240 samples
  3 predictor

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 240, 240, 240, 240, 240, 240, ... 
Resampling results across tuning parameters:

  C     RMSE      Rsquared   MAE     
  0.25  1221.831  0.3597043  274.8764
  0.50  1199.844  0.3825125  257.0950
  1.00  1178.357  0.4112424  246.6976

Tuning parameter 'sigma' was held constant at a value of 0.7220883
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.7220883 and C = 1.
"

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$biomassa, predicoes.svm)
"
[1] 406.4227
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$biomassa, predicoes.svm)
"
[1] 0.8281087
"

# Cross-validation SVM
ctrl <- trainControl(method = "cv", number = 10)

set.seed(37)
svm <- train(biomassa~., data=treino, method="svmRadial", trControl=ctrl)
svm
"
Support Vector Machines with Radial Basis Function Kernel 

240 samples
  3 predictor

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 216, 216, 216, 216, 216, 216, ... 
Resampling results across tuning parameters:

  C     RMSE      Rsquared   MAE     
  0.25  860.8728  0.6649424  278.6009
  0.50  842.0033  0.6793715  262.3108
  1.00  828.0175  0.6790947  249.4843

Tuning parameter 'sigma' was held constant at a value of 0.7220883
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.7220883 and C = 1.
"

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$biomassa, predicoes.svm)
"
[1] 406.4227
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$biomassa, predicoes.svm)
"
[1] 0.8281087
"

# Vários C e sigma
tuneGrid = expand.grid(C=c(1, 2, 10, 50, 100), sigma=c(.01, .015, 0.2))

set.seed(37)
svm <- train(biomassa~., data=treino, method="svmRadial", trControl=ctrl, tuneGrid=tuneGrid)
svm
"
Support Vector Machines with Radial Basis Function Kernel 

240 samples
  3 predictor

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 216, 216, 216, 216, 216, 216, ... 
Resampling results across tuning parameters:

  C    sigma  RMSE      Rsquared   MAE     
    1  0.010  674.5961  0.8530830  220.4585
    1  0.015  654.5896  0.8510234  215.5155
    1  0.200  772.2547  0.7496164  229.8705
    2  0.010  599.6565  0.8740859  204.9682
    2  0.015  590.6644  0.8665970  207.0339
    2  0.200  729.0287  0.7782850  219.9035
   10  0.010  394.8024  0.8820733  152.8290
   10  0.015  389.0714  0.8781829  141.1168
   10  0.200  617.6300  0.8073477  204.6920
   50  0.010  288.8744  0.8928991  112.9204
   50  0.015  333.4633  0.8956423  125.2427
   50  0.200  611.6631  0.8167221  202.3063
  100  0.010  282.6566  0.8938355  111.4711
  100  0.015  337.8377  0.9017036  149.1986
  100  0.200  610.7222  0.8204359  201.0111

RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.01 and C = 100.
"

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Calcular as métricas
rmse(teste$biomassa, predicoes.svm)
"
[1] 308.6657
"

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((predito-mean(observado))^2)))
}
r2(teste$biomassa, predicoes.svm)
"
[1] 0.9018723
"

# Predições de novos casos
dados_novos_casos <- read.csv("https://raw.githubusercontent.com/marcos-bezerra/MachineLearning/main/Biomassa_Regressao/biomassa_dados_novos_casos.csv")
dados_novos_casos
"
   dap    h   Me biomassa
1  6.4  7.0 1.04        ?
2  7.3 10.0 1.04        ?
3  7.8  5.5 1.04        ?
4 12.2  7.5 1.04        ?
"

dados_novos_casos$biomassa <- NULL
predict.svm <- predict(svm, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predict.svm)
resultado
"
   dap    h   Me predict.svm
1  6.4  7.0 1.04    159.2163
2  7.3 10.0 1.04    150.5539
3  7.8  5.5 1.04    165.1988
4 12.2  7.5 1.04    157.8682
"

