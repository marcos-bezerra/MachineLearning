setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

# instalar bibliotecas
install.packages("caret")
install.packages("e1071")
install.packages("Metrics")
install.packages("mlbench")
install.packages("mice")

# carregando as bibliotecas
library(Metrics)
library(mlbench)
library(caret)
library(mice)

# carregando a base de dados
dados <- read.csv("https://raw.githubusercontent.com/marcos-bezerra/MachineLearning/main/Admissao_Regressao/Admissao_Dados.csv")

dim(dados)
summary(dados)
View(dados)

# tratando valores missing
dados$Serial.No. <- NULL
#imp <- mice(dados) # pacote mice trata dados ausentes, inputa os dados ausentes "m" vezes
#dados <- complete(imp,1) # exporta dados imputados

# Cria arquivo de treino e teste
set.seed(37)
indices <- createDataPartition(dados$ChanceOfAdmit, p=0.80, list=FALSE) 
treino <- dados[indices,]
teste <- dados[-indices,]


##############################                   ########################
#                             Funções - Regressão                       #
##############################                   ########################

# Regresão RMSE - raiz quadrada do erro médio
rmse_ <- function(valor_real, valor_estimado) {
  return(sqrt(sum((valor_real-valor_estimado)^2) / nrow(teste)))
}

# Regressão R2 - Coeficiente de Determinação Múltipla
r2 <- function(valor_real, valor_estimado) {
  return(1 - (sum((valor_real - valor_estimado)^2) /
                sum((valor_real - mean(valor_real))^2)))
}
  
# Regressão Syx - Erro padrão de Estimativa
syx <- function(valor_real, valor_estimado) {
  return(sum((valor_real-valor_estimado)^2) /
           (nrow(teste)-ncol(teste)))
}

# Regressão Pearson com CV
pearson <- function(valor_real, valor_estimado) {
  return(
    sum(
      (valor_real-mean(valor_real)) * (valor_estimado-mean(valor_estimado))
    ) /
      (
        sqrt(sum(valor_real-mean(valor_real))^2)*
          sqrt(sum(valor_estimado-mean(valor_estimado))^2)
      )
  )
}

# Regressão MAE - Média Absoluta do erro
mae <- function(valor_real, valor_estimado) {
  return(sum(abs(valor_real-valor_estimado)) / nrow(teste))
}


##############################                 ##########################
#                              RNA - Regressão                          #
##############################                 ##########################

# Treinar o modelo com Hold-Out
set.seed(37)
rna_hold_out <- train(ChanceOfAdmit~., data=treino, method="nnet", linout=TRUE, trace=FALSE)
rna_hold_out
"
Neural Network 

402 samples
  7 predictor

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 402, 402, 402, 402, 402, 402, ... 
Resampling results across tuning parameters:

  size  decay  RMSE        Rsquared   MAE       
  1     0e+00  0.13471187  0.5496340  0.10917622
  1     1e-04  0.13136178  0.8223088  0.10618851
  1     1e-01  0.07076205  0.7513540  0.05341829
  3     0e+00  0.12693528  0.6827204  0.10158023
  3     1e-04  0.08312890  0.7091376  0.06346781
  3     1e-01  0.07049393  0.7535470  0.05341777
  5     0e+00  0.13753332  0.4225692  0.11118365
  5     1e-04  0.07953995  0.6895798  0.06085130
  5     1e-01  0.06863512  0.7660235  0.05191373

RMSE was used to select the optimal model using the smallest value.
The final values used for the model were size = 5 and decay = 0.1.
"

# Calculo das predições para a base teste
predicoes.rna_hold_out <- predict(rna_hold_out, teste)

# Cálculo das métricas Metrics - (rmse)
rmse(teste$ChanceOfAdmit, predicoes.rna_hold_out)
"
[1] 0.07313229
"

# Regresão RMSE - raiz quadrada do erro médio
rmse_(teste$ChanceOfAdmit, predicoes.rna_hold_out)
"
[1] 0.07313229
"

# Regressão R2 - Coeficiente de Determinação Múltipla
r2(teste$ChanceOfAdmit, predicoes.rna_hold_out)
"
[1] 0.7404621
"

# Regressão Syx - Erro padrão de Estimativa
syx(teste$ChanceOfAdmit, predicoes.rna_hold_out)
"
[1] 0.005823738
"

# Regressão Pearson com CV
pearson(teste$ChanceOfAdmit, predicoes.rna_hold_out)
"
[1] 1.68825e+29
"

# Regressão MAE - Média Absoluta do erro
mae(teste$ChanceOfAdmit, predicoes.rna_hold_out)
"
[1] 0.05233998
"

#.........................................................................

# Cross Validation e parametrização da RNA
control <- trainControl(method = "cv", number = 10)
tuneGrid <- expand.grid(size = seq(from = 1, to = 10, by = 1),
                        decay = seq(from = 0.1, to = 0.9, by = 0.3))

set.seed(37)
rna_cv <- train(ChanceOfAdmit~., data=treino, method="nnet", trainControl=control,
             tuneGrid=tuneGrid, linout=TRUE, MaxNWts=10000, maxit=2000, trace=FALSE)
rna_cv
"
Neural Network 

402 samples
  7 predictor

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 402, 402, 402, 402, 402, 402, ... 
Resampling results across tuning parameters:

  size  decay  RMSE        Rsquared   MAE       
   1    0.1    0.07086613  0.7507427  0.05352238
   1    0.4    0.07975728  0.6835294  0.06127064
   1    0.7    0.08127231  0.6722683  0.06283097
   2    0.1    0.06971844  0.7581175  0.05278818
   2    0.4    0.07838632  0.6943109  0.06003757
   2    0.7    0.08035509  0.6792397  0.06189665
   3    0.1    0.06906360  0.7631089  0.05213889
   3    0.4    0.07734505  0.7026822  0.05916813
   3    0.7    0.07960270  0.6855803  0.06120235
   4    0.1    0.06774301  0.7725393  0.05113195
   4    0.4    0.07637868  0.7103203  0.05836128
   4    0.7    0.07868842  0.6934983  0.06048692
   5    0.1    0.06728285  0.7750893  0.05067231
   5    0.4    0.07557533  0.7168999  0.05775395
   5    0.7    0.07809508  0.6982962  0.05995993
   6    0.1    0.06657732  0.7800492  0.05005906
   6    0.4    0.07479202  0.7226524  0.05707801
   6    0.7    0.07720179  0.7055655  0.05923822
   7    0.1    0.06573453  0.7856080  0.04937178
   7    0.4    0.07402303  0.7286139  0.05648668
   7    0.7    0.07680211  0.7086985  0.05890632
   8    0.1    0.06524871  0.7890517  0.04889954
   8    0.4    0.07307232  0.7359549  0.05569307
   8    0.7    0.07591830  0.7159546  0.05821033
   9    0.1    0.06453599  0.7934911  0.04825061
   9    0.4    0.07243231  0.7408864  0.05517285
   9    0.7    0.07529781  0.7209519  0.05770860
  10    0.1    0.06453220  0.7934540  0.04828197
  10    0.4    0.07184767  0.7446331  0.05471869
  10    0.7    0.07475167  0.7250963  0.05727537

RMSE was used to select the optimal model using the smallest value.
The final values used for the model were size = 10 and decay = 0.1.
"

# Predições e métricas aplicadas na base teste
predicoes.rna_cv <- predict(rna_cv, teste)

# Cálculo das métricas Metrics - (rmse)
rmse(teste$ChanceOfAdmit, predicoes.rna_cv)
"
[1] 0.06824529
"

# Regresão RMSE - raiz quadrada do erro médio
rmse_(teste$ChanceOfAdmit, predicoes.rna_cv)
"
[1] 0.06824529
"

# Regressão R2 - Coeficiente de Determinação Múltipla
r2(teste$ChanceOfAdmit, predicoes.rna_cv)
"
[1] 0.7739899
"

#Regressão Syx com CV
syx(teste$ChanceOfAdmit, predicoes.rna_cv)
"
[1] 0.005071412
"

#Regressão Pearson com CV
pearson(teste$ChanceOfAdmit, predicoes.rna_cv)
"
[1] 2.623262e+29
"

#Regressão MAE - Média Absoluta do erro
mae(teste$ChanceOfAdmit, predicoes.rna_cv)
"
[1] 0.04677909
"

#.........................................................................

# Executando o modelo com os melhores hiperparâmetros
grid <- expand.grid(size = c(10), decay =c(0.1))

set.seed(37)
rna_melhor_modelo <- train(ChanceOfAdmit ~.,
                           data = treino,
                           method = "nnet",
                           trainControl = control,
                           tuneGrid = grid,
                           linout=TRUE,
                           MaxNWts=10000,
                           maxit = 2000,
                           trace=FALSE) 
rna_melhor_modelo
"
Neural Network 

402 samples
  7 predictor

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 402, 402, 402, 402, 402, 402, ... 
Resampling results:

  RMSE        Rsquared   MAE       
  0.06410871  0.7960591  0.04784236

Tuning parameter 'size' was held constant at a value of 10

Tuning parameter 'decay' was held constant at a value of 0.1
"

# Predições e métricas aplicadas na base teste
predicoes.rna_melhor_modelo <- predict(rna_melhor_modelo, teste)

# Cálculo das métricas Metrics - (rmse)
rmse(teste$ChanceOfAdmit, predicoes.rna_melhor_modelo)
"
[1] 0.06923073
"

# Regresão RMSE - raiz quadrada do erro médio
rmse_(teste$ChanceOfAdmit, predicoes.rna_melhor_modelo)
"
[1] 0.06923073
"

# Regressão R2 - Coeficiente de Determinação Múltipla
r2(teste$ChanceOfAdmit, predicoes.rna_melhor_modelo)
"
[1] 0.7674158
"

#Regressão Syx com CV
syx(teste$ChanceOfAdmit, predicoes.rna_melhor_modelo)
"
[1] 0.005218928
"

#Regressão Pearson com CV
pearson(teste$ChanceOfAdmit, predicoes.rna_melhor_modelo)
"
[1] 1.335397e+29
"

#Regressão MAE - Média Absoluta do erro
mae(teste$ChanceOfAdmit, predicoes.rna_melhor_modelo)
"
[1] 0.04814263
"

#.........................................................................

# Salvar o melhor modelo
saveRDS(melhor_modelo_rna, "admissao_melhor_modelo_rna.rds")

#.........................................................................

# Aplicar o Melhor Modelo em predições de novos casos
dados_novos_casos <- read.csv("https://raw.githubusercontent.com/marcos-bezerra/MachineLearning/main/Admissao_Regressao/Admissao_Dados_Novos_Casos.csv")
View(dados_novos_casos)
dim(dados_novos_casos)

dados_novos_casos$Serial.No. <- NULL

melhor_modelo <- readRDS("admissao_melhor_modelo_rna.rds")
predicoes.rna <- predict(melhor_modelo, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predicoes.rna)
View(resultado)

#................................... Fim .................................


##############################                 ##########################
#                              KNN - Regressão                          #
##############################                 ##########################

# Prepara um grid com os valores de k que serão usados 
tuneGrid <- expand.grid(k = c(1,3,5,7,9))

# Executa o Knn com o grid criado
set.seed(37)
knn <- train(ChanceOfAdmit ~ ., data = treino, method = "knn", tuneGrid=tuneGrid)
knn

"
k-Nearest Neighbors 

402 samples
  7 predictor

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 402, 402, 402, 402, 402, 402, ... 
Resampling results across tuning parameters:

  k  RMSE        Rsquared   MAE       
  1  0.09641199  0.5879902  0.07101872
  3  0.08210127  0.6760871  0.06149820
  5  0.07641231  0.7125685  0.05744375
  7  0.07409833  0.7278781  0.05574942
  9  0.07279440  0.7364567  0.05502333

RMSE was used to select the optimal model using the smallest value.
The final value used for the model was k = 9.
"

# Aplica o modelo no arquivo de teste
predicoes.knn <- predict(knn, teste)

# Cálculo das métricas Metrics - (rmse)
rmse(teste$ChanceOfAdmit, predicoes.knn)
"
[1] 0.08384332
"

# Regresão RMSE - raiz quadrada do erro médio
rmse_(teste$ChanceOfAdmit, predicoes.knn)
"
[1] 0.08384332
"

# Regressão R2 - Coeficiente de Determinação Múltipla
r2(teste$ChanceOfAdmit, predicoes.knn)
"
[1] 0.6588849
"

# Regressão Syx - Erro padrão de Estimativa
syx(teste$ChanceOfAdmit, predicoes.knn)
"
[1] 0.007654565
"

# Regressão Pearson com CV
pearson(teste$ChanceOfAdmit, predicoes.knn)
"
[1] 1.370892e+29
"

# Regressão MAE - Média Absoluta do erro
mae(teste$ChanceOfAdmit, predicoes.knn)
"
[1] 0.06084354
"

# Regressão - Gráfico de Resíduos
vlr_real = teste$ChanceOfAdmit
vlr_predito = predicoes.knn
vlr_residuos <- (vlr_real - vlr_predito)/vlr_real*100

plot(
  x = teste$ChanceOfAdmit, 
  y = vlr_residuos,
  main= "Gráfico de Resíduos",
  xlab = "Predições",
  ylab = "% Residuos",
  pch= 42,
  col= 3,
  cex.main = 1.75,
  col.axis= 8,
  cex.axis = 1,
  col.lab= 9,
  cex.lab = 1.25,
  abline(0,0)
)
grid()

#.........................................................................

# Predições de novos casos
dados_novos_casos <- read.csv("https://raw.githubusercontent.com/marcos-bezerra/MachineLearning/main/Admissao_Regressao/Admissao_Dados_Novos_Casos.csv")
View(dados_novos_casos)

dados_novos_casos$Serial.No. <- NULL
predicoes.knn <- predict(knn, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predicoes.knn)
View(resultado)


##############################                 ##########################
#                              SVM - Regressão                          #
##############################                 ##########################

# Treinar SVM com a base de Treino
set.seed(37)
svm <- train(ChanceOfAdmit~., data=treino, method="svmRadial") 
svm

"
Support Vector Machines with Radial Basis Function Kernel 

402 samples
  7 predictor

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 402, 402, 402, 402, 402, 402, ... 
Resampling results across tuning parameters:

  C     RMSE        Rsquared   MAE       
  0.25  0.06526839  0.7936945  0.04789075
  0.50  0.06441184  0.7960826  0.04717612
  1.00  0.06480668  0.7922980  0.04744039

Tuning parameter 'sigma' was held constant at a value of 0.1894597
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.1894597 and C = 0.5.
"
# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Cálculo das métricas Metrics - (rmse)
rmse(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.07006415
"

# Regresão RMSE - raiz quadrada do erro médio
rmse_(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.07006415
"

# Regressão R2 - Coeficiente de Determinação Múltipla
r2(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.7617822
"

# Regressão Syx - Erro padrão de Estimativa
syx(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.00534534
"

# Regressão Pearson com CV
pearson(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 1.592568e+29
"

# Regressão MAE - Média Absoluta do erro
mae(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.04611024
"

#.........................................................................

# Cross-validation SVM
ctrl <- trainControl(method = "cv", number = 10)

set.seed(37)
svm <- train(ChanceOfAdmit~., data=treino, method="svmRadial", trControl=ctrl)
svm
"
Support Vector Machines with Radial Basis Function Kernel 

402 samples
  7 predictor

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 362, 362, 362, 362, 362, 362, ... 
Resampling results across tuning parameters:

  C     RMSE        Rsquared   MAE       
  0.25  0.06428039  0.8018458  0.04657333
  0.50  0.06295222  0.8052478  0.04541705
  1.00  0.06170558  0.8089594  0.04460387

Tuning parameter 'sigma' was held constant at a value of 0.1894597
RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.1894597 and C = 1.
"

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Cálculo das métricas Metrics - (rmse)
rmse(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.06953861
"

# Regresão RMSE - raiz quadrada do erro médio
rmse_(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.06953861
"

# Regressão R2 - Coeficiente de Determinação Múltipla
r2(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.7653425
"

#Regressão Syx com CV
syx(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.005265451
"

#Regressão Pearson com CV
pearson(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 1.676159e+29
"

#Regressão MAE - Média Absoluta do erro
mae(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.04513253
"

#.........................................................................

# Vários C e sigma
tuneGrid = expand.grid(C=c(1, 2, 10, 50, 100), sigma=c(.01, .015, 0.2))

set.seed(37)
svm <- train(ChanceOfAdmit~., data=treino, method="svmRadial", trControl=ctrl, tuneGrid=tuneGrid)
svm
"
Support Vector Machines with Radial Basis Function Kernel 

402 samples
  7 predictor

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 362, 362, 362, 362, 362, 362, ... 
Resampling results across tuning parameters:

  C    sigma  RMSE        Rsquared   MAE       
    1  0.010  0.05974872  0.8245636  0.04243110
    1  0.015  0.05972865  0.8236084  0.04262774
    1  0.200  0.06190299  0.8078503  0.04481324
    2  0.010  0.05934410  0.8249471  0.04221667
    2  0.015  0.05947930  0.8235208  0.04231560
    2  0.200  0.06241391  0.8031725  0.04525250
   10  0.010  0.05910425  0.8242834  0.04205354
   10  0.015  0.05924052  0.8229045  0.04218353
   10  0.200  0.07052160  0.7498313  0.05135221
   50  0.010  0.05912247  0.8225276  0.04181544
   50  0.015  0.05974111  0.8183938  0.04221256
   50  0.200  0.08316789  0.6771144  0.06029146
  100  0.010  0.05982342  0.8179276  0.04227564
  100  0.015  0.05973165  0.8181600  0.04219274
  100  0.200  0.09321473  0.6202139  0.06761342

RMSE was used to select the optimal model using the smallest value.
The final values used for the model were sigma = 0.01 and C = 10.
"

# Aplicar modelos treinados na base de Teste
predicoes.svm <- predict(svm, teste)

# Cálculo das métricas Metrics - (rmse)
rmse(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.06683756
"

# Regresão RMSE - raiz quadrada do erro médio
rmse_(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.06683756
"

# Regressão R2 - Coeficiente de Determinação Múltipla
r2(teste$ChanceOfAdmit, predicoes.svm)
"
0.7832178
"

#Regressão Syx com CV
syx(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.004864349
"

#Regressão Pearson com CV
pearson(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 7.905453e+29
"

#Regressão MAE - Média Absoluta do erro
mae(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.04308562
"

#.........................................................................

# Executando o modelo com os melhores hiperparâmetros
grid <- expand.grid(size = c(1), decay =c(0.1894597))

set.seed(37)
melhor_modelo_svm <- train(ChanceOfAdmit ~.,
                           data = treino,
                           method = "nnet",
                           trainControl = control,
                           tuneGrid = grid,
                           linout=TRUE,
                           MaxNWts=10000,
                           maxit = 2000,
                           trace=FALSE) 
melhor_modelo_svm
"
Neural Network 

402 samples
  7 predictor

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 402, 402, 402, 402, 402, 402, ... 
Resampling results:

  RMSE        Rsquared   MAE       
  0.07717116  0.7033269  0.05877439

Tuning parameter 'size' was held constant at a value of 1

Tuning parameter 'decay' was held constant at a value
 of 0.1894597
"

# Predições e métricas aplicadas na base teste
predicoes.svm <- predict(melhor_modelo_svm, teste)

# Cálculo das métricas Metrics - (rmse)
rmse(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.08069547
"

# Regresão RMSE - raiz quadrada do erro médio
rmse_(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.08069547
"

# Regressão R2 - Coeficiente de Determinação Múltipla
r2(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.6840046
"

#Regressão Syx com CV
syx(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.007090582
"

#Regressão Pearson com CV
pearson(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 1.617527e+30
"

#Regressão MAE - Média Absoluta do erro
mae(teste$ChanceOfAdmit, predicoes.svm)
"
[1] 0.058217
"

#.........................................................................

# Salvar o melhor modelo
saveRDS(melhor_modelo_svm, "admissao_melhor_modelo_svm.rds")

#.........................................................................

# Aplicar o Melhor Modelo em predições de novos casos
dados_novos_casos <- read.csv("https://raw.githubusercontent.com/marcos-bezerra/MachineLearning/main/Admissao_Regressao/Admissao_Dados_Novos_Casos.csv")
View(dados_novos_casos)
dim(dados_novos_casos)

dados_novos_casos$Serial.No. <- NULL

melhor_modelo <- readRDS("admissao_melhor_modelo_svm.rds")
predicoes.svm <- predict(melhor_modelo, dados_novos_casos)
resultado <- cbind(dados_novos_casos, predicoes.svm)
View(resultado)

#.........................................................................

# Gráfico de Resíduos da técnica com parâmetro de maior R2
vlr_real = teste$ChanceOfAdmit
vlr_predito = predicoes.rna
vlr_residuos <- (vlr_real - vlr_predito)/vlr_real*100

plot(
  x = teste$ChanceOfAdmit, 
  y = vlr_residuos,
  main= "Gráfico de Resíduos",
  xlab = "Predições",
  ylab = "% Residuos",
  pch= 42,
  col= 3,
  cex.main = 1.75,
  col.axis= 8,
  cex.axis = 1,
  col.lab= 9,
  cex.lab = 1.25,
  abline(0,0)
  )
grid()

