import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split


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