import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split


def custom_cost_metric (y_actual, y_pred):

    """
    Função personalizada para usar como métrica de seleção no treino dos modelos usando o Gridsearch
    
     """

    TN , FP , FN , TP = confusion_matrix(y_actual, y_pred).ravel().tolist()

    cost = FP + 10*FN 

    return cost




def metrics_AUROC_AUPR(df, label, probability):

    """
    Retorna as métricas de área baixo as curvas ROC e PR:

        df:= DataFrame pandas escorado
        label:= variáivel resposta (label). 0 Para não evento e 1 para evento
        Probability:= Score ou probabilidade extraída de um modelo.
    
    """

    fpr, tpr, thresholds_roc =  roc_curve(y_true = df[label], y_score = df[probability] )
    AUROC = auc(fpr,tpr)

    precision, recall, thresholds_pr = precision_recall_curve(y_true = df[label], probas_pred = df[probability] )
    AUPR = auc(recall, precision)

    return AUROC , AUPR



def train_test_split_2_model(df, label, test_size, random_state, by_date = False, date_field = "data", n_days_test = 5 ):

    """"
    Retorna um X_train, X_test, y_train, y_test 
    
        df:= DataFrame pandas escorado
        label:= variáivel resposta (label). 0 Para não evento e 1 para evento
        test_size:= Tamanho da amostra para o conjunto de teste.
        random_state:= Semente para realizar a divisão do conjunto. 
        by_date:= True para quando deseje dividir o conjunto usando o campo de data.
        date_field:= Nome do campo de data, em formato de data.
        n_days_test:= Número de dias para selecionar o conjunto de teste.
    
    """

    if by_date == False:

        X = df.drop([label,date_field], axis=1)
        y = df[label]

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=random_state, stratify = y )
    
    else:

        n = n_days_test
        ## Selecting the train 
        datas_train = np.sort( df[date_field].unique() )[:-n]
        df_train = df[df[date_field].isin(datas_train)]

        X_train = df_train.drop([label,date_field], axis=1)
        y_train = df_train[label]


        ## selecting the last n dates to validate (aprox 14% of full dataset)
        datas_test = np.sort( df[date_field].unique() )[::-1][:n]
        df_test = df[df[date_field].isin(datas_test)]

        X_test = df_test.drop([label,date_field], axis=1)
        y_test = df_test[label]

        

    return X_train, X_test, y_train, y_test


def cross_tab_matrices( df, label = "fraude", probability = "score", ticket = 'monto', cutoff = 0, total = False ):

    """
    Retorna a matriz de confusão e matriz de custo

        df:=
        label:=
        probability:=
        ticket:=
        cutoff:-
        total:=
    
    
    """

    df_aval = df[ [label, probability, ticket] ]
    df_aval['predict'] = df_aval[probability].apply( lambda x: 1 if x>=cutoff else 0  )

    confusion_matrix = pd.crosstab( df_aval[label], df_aval['predict'], margins= total  )


    confusion_matrix_ticket = pd.crosstab( df_aval[label], df_aval['predict'], values = df[ticket], aggfunc="sum" , margins= total) 


    return confusion_matrix, confusion_matrix_ticket


def confusion_matrix_2_TP_FP_TN_FN(confusion_matrix, total =False):

    """
    Retorno as valores de TP, FP, TN, FN da matriz de confusão 
    
        confusion_matrix:= Matriz de confusão 
        total:= True para cria coluna de totais


    """


    try:
        TP = confusion_matrix.iloc[lambda x: x.index == 1, lambda x: x.columns == 1][1][1]
    except:
        TP = 0

    try:
        FP = confusion_matrix.iloc[lambda x: x.index == 0, lambda x: x.columns == 1][1][0]
    except: 
        FP = 0

    try:
        TN = confusion_matrix.iloc[lambda x: x.index == 0, lambda x: x.columns == 0][0][0]
    except: 
        TN = 0

    try:
        FN = confusion_matrix.iloc[lambda x: x.index == 1, lambda x: x.columns == 0][0][1]
    except:
        FN = 0

    ## Computing the events and non events
    events = confusion_matrix.iloc[lambda x: x.index == 1].sum().sum()
    non_events = confusion_matrix.iloc[lambda x: x.index == 0].sum().sum()

    ## If total is True, we need to divide by 2
    P = events/2 if total == True else events
    N = non_events/2 if total == True else non_events


    return TP, FP, TN, FN, P, N


def confusion_matrix_2_bussines(confusion_matrix_ticket, total =False):

    """
    Retorno os valores da matriz de confusão

        confusion_matrix:= Matriz de confusão   
        total:= True para cria coluna de totais

    gain:= Total de vendas de compras que foram catalogadas como não fraude pelo modelo e ram compras legitimas. TN
    lost_by_bloq:= Total de vendas perdidas por ser catalogadas como fraude pelo modelo e que eram compras legitimas. FP  
    lost_by_fraud:= Total de vendas perdidas por ser catalogas como não fraude pelo modelo e que eram compras fraudolentas. FN
    success_by_fraud:= Total de vendas que foram catalogadas como fraude pelo modelo e que foram fraudolentas. TP

    """

    try:
        gain = confusion_matrix_ticket.iloc[lambda x: x.index == 0, lambda x: x.columns == 0][0][0]
    except:
        gain = 0

    try:
        lost_by_bloq = confusion_matrix_ticket.iloc[lambda x: x.index == 0, lambda x: x.columns == 1][1][0]
    except:
        lost_by_bloq = 0

    try:
        lost_by_fraud = confusion_matrix_ticket.iloc[lambda x: x.index == 1, lambda x: x.columns == 0][0][1]
    except:
        lost_by_fraud = 0

    try:
        success_by_fraud = confusion_matrix_ticket.iloc[lambda x: x.index == 1, lambda x: x.columns == 1][1][1]
    except:
        success_by_fraud = 0


    total_ticket = confusion_matrix_ticket.sum().sum()
    
    ## We need to dividy by 4 when total is true.
    total_sell = total_ticket/4 if total==True else total_ticket


    return gain, lost_by_bloq, lost_by_fraud, success_by_fraud, total_sell


def gain_value(df, score, label = "fraude", probability = "score", ticket = "monto"  ):

    """
    Retorna todas os valores da matriz de confusão além das métricas de modelo e de negócio

        df:= Datatrame com a coluna de score
        score:= Valor do score para o cálculo
        label:= Nome da variável resposta com o valores de 1 (positivos) para evento e 0(Nagativos) para não evento.
        probability:= Nome da coluna score
        ticket:= Nome da coluna de valor da compra
    """

    ## Calculando a matriz de confusão e de negócio
    confusion_matrix, confusion_matrix_monto = cross_tab_matrices(df, label , probability , ticket , cutoff = score, total = False)

    ######################### Métricas ##################################################
    ## Extraindo os valores da matriz de confusão
    TP, FP, TN, FN, P, N = confusion_matrix_2_TP_FP_TN_FN( confusion_matrix, total = False  )
    gain, lost_by_bloq, lost_by_fraud, success_by_fraud, total_sell = confusion_matrix_2_bussines(confusion_matrix_monto, total = False)

    ## Métricas de performance
    recall = TP/P
    FPR = FP/(FP+TP)

    ## Métricas de negócio
    ganho_com_bloq = (0.1*gain-lost_by_fraud-0.1*lost_by_bloq)
    ganho_sem_bloq = (0.1*gain-lost_by_fraud)

    return TP, FP, TN, FN, P, N, gain, lost_by_bloq, lost_by_fraud, success_by_fraud, total_sell, recall, FPR, ganho_com_bloq, ganho_sem_bloq