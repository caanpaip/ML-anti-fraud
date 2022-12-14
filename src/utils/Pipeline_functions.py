import joblib
import sys
import os
import pandas as pd

if sys.platform=='win32':
    sys.path.insert(0,".\\")
  
elif sys.platform=='linux':
    sys.path.insert(0,"./")
 

from utils import utils_ml, parameters

############################################ Criação de colunas Bins #############################################

def create_woe_columns(df, path_pickle_transf ):

    """ 
    Retorna um dataframe do pandas com colunas adicionáis dada pela tranformação WOE e BINS do para o modelo.

        df:= Dataframe com os dados 
        path_pickle_transf:= Caminho do pickle que tem as tranformações WOE e BIN por cada feature.
    
    
    """

    ## criando uma copia
    dff = df.copy()
    ## Selecionando as colunas para serem usadas
    colunas_orig_and_week = utils_ml.exclude_words( dff.columns.tolist(), ["cat","bin","WoE","produto","fecha","data","hora","week","fraude"] ) + ['categoria_produto']
    ## Iterando para adicionar as colunas WoE
    for variable in colunas_orig_and_week:

        pkl_transform_feature = os.path.join( path_pickle_transf,   f"pipeline_binning__{variable}__.pkl") 

        optb = joblib.load( pkl_transform_feature  )

        dff = utils_ml.encode_woe_var_transform(dff, variable, optb )


    return dff

################################################ REGRESSÃO LOGISTICA ###################################################

def df_features_bins(df, path_pickle_transf ):

    """ 
    Retorna um dataframe do pandas com colunas que tem os bins de cada feature.

        df:= Dataframe com os dados.
        path_pickle_transf:= Caminho do pickle que tem as tranformações WOE e BIN por cada feature.
    
    
    """    

    df_woe_bins_features = create_woe_columns(df, path_pickle_transf )

    features_bins_woe = utils_ml.list_subset_words( df_woe_bins_features.columns.tolist(), ["bin"])
    ## Excluding variable 'score' (score of Meli model) and 'o_obj' (with 72.5% of missing values))
    features_bins_woe = utils_ml.exclude_words(features_bins_woe, ["score","o_obj"])
    
    try:
        df_bins = df_woe_bins_features[features_bins_woe + ["fraude"]]
    except Exception as e:

        print(e)

        df_bins = df_woe_bins_features[features_bins_woe]

    return df_bins


def transform_2_dummies( df , label = "fraude", drop_first = True):

    """
    Retorno o dataframe com as variáveis dummies para o modelo.

        df:= Dataframe com as variáveis com prefixo bins
        label:= Variável resposta
        drop_first:= Elimina as dummies redundantes

    
    """

    features = utils_ml.list_subset_words( df.columns.tolist(), ["bin"])
    features = utils_ml.exclude_words(features, ["score","o_obj"])

    ls_df_dummies = []

    for var in features:

        ## creating dummy to variable
        df_dummy = pd.get_dummies( df[var] , prefix=var, drop_first = drop_first)
        ls_df_dummies.append(df_dummy)

    try:
        df_2_model_dum = pd.concat(ls_df_dummies + [ df[label] ], axis=1)
    except Exception as e:

        print(e)

        df_2_model_dum = pd.concat(ls_df_dummies , axis=1)

    return df_2_model_dum



def transform_2_model(df ,  label ="fraude", cols_model = parameters.cols_rg_model ):

    """
    Retorna o dataframe de entrada do modelo de regressão logísticas.transform_2_dummies

        df:= Dataframe com os campos com prefixo bin
        label:= Variável resposta
        col_modes:= Campos usados no treinamento do modelo

    """

    ## tirando a coluna label caso esteja na lista
    cols_model = utils_ml.exclude_words(cols_model, [label] )
    ## Transformando para dummies
    df_marco_model_dum = transform_2_dummies(df = df, label = label)

    ## Preenchendo colunas faltantes
    dummies_faltantes = list(   set(cols_model) - set(df_marco_model_dum.columns.to_list()) )

    if len(dummies_faltantes)>0:

        for dummi_falt in dummies_faltantes:

            df_marco_model_dum[dummi_falt] = 0

    try:

        df_dummies_features = df_marco_model_dum[cols_model+[label]]

    except Exception as e:

        print(e)

        df_dummies_features = df_marco_model_dum[cols_model]

    return df_dummies_features


def scoring_with_lg(df, path_pickle_transf = parameters.features_path, path_model = parameters.model_path  ):
    """
    Retorna o dataframe com a scoragem feita pela regressão logistica treinada

        df:= Dataframe do pandas com os dados de entrada do modelo.
        path_pickle_transf:= Caminho onde encontra-se o pkl para tranformas as variáveis em bins.
        path_model:= Caminho onde encontra-se o pkl do modelo da regressão logística.

    """

    df_ = df.copy()


    ## bins
    df_bins_lg = df_features_bins(df_, path_pickle_transf )
    ## df 2 model
    df_2_model = transform_2_model(df_bins_lg ,  label ="fraude", cols_model = parameters.cols_rg_model )
    ## Carregando o modelo
    model_lg = joblib.load( os.path.join( path_model , f"model_lg.pkl") )
    ## Colocando o score
    df_['probability'] = model_lg.predict_proba(df_2_model)[:,1]

    return df_

##################################################### CATBOOST ################################################################


def df_features_cat(df, path_pickle_transf ):

    """ 
    Retorna um dataframe do pandas com colunas que tem as categorias de cada feature.

        df:= Dataframe com os dados.
        path_pickle_transf:= Caminho do pickle que tem as tranformações WOE e BIN por cada feature.
    
    
    """    

    df_woe_bins_features = create_woe_columns(df, path_pickle_transf )

    ## Selecionando as colunas de categorias criadas pelo tranformador anterior
    cat_features = [x for x in utils_ml.list_subset_words( df_woe_bins_features.columns.tolist(), ['cat']) if "WoE" not in x and "bin" not in x and x!='categoria_produto']
    ## Excluindo as colunas de score, o_obj e data caso existam
    cat_features = utils_ml.exclude_words(cat_features, ["score","o_obj","data"])
    
    try:
        df_2_model_cat = df_woe_bins_features[ cat_features + ["fraude"] ]
    except Exception as e:
        print(e)
        df_2_model_cat = df_woe_bins_features[ cat_features ]


    return df_2_model_cat

def scoring_with_cat( df , path_pickle_transf = parameters.features_path , path_model = parameters.model_path ):

    df_ = df.copy()
    ## tranformando e selecionando variáveis do modelo do catboost
    df_cat = df_features_cat( df_, path_pickle_transf )
    ## chamando o modelo do catboost salvo no pkl
    modelo_cat = joblib.load( os.path.join( path_model , f"model_cat.pkl") )
    # scorando usando o cat model
    df_['probability'] = modelo_cat.predict_proba(df_cat)[:,1]

    return df_