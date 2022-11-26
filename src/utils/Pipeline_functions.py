import sys
import joblib

if sys.platform=='win32':
    sys.path.insert(0,".\\")
    features_path = ".\..\src\\features"
        
elif sys.platform=='linux':
    sys.path.insert(0,"./")
    features_path = "./../src/features"



from utils import utils_ml, parameters, metrics_evaluation


def create_woe_columns(df):

    ## criando uma copia
    dff = df.copy()
    ## Selecionando as colunas para serem usadas
    colunas_orig_and_week = utils_ml.exclude_words( dff.columns.tolist(), ["cat","bin","WoE","produto","fecha","data","hora","week","fraude"] ) + ['categoria_produto']
    ## Iterando para adicionar as colunas WoE
    for n,variable in enumerate(colunas_orig_and_week, start=1):

        optb = joblib.load(f"{features_path}\\pipeline_binning__{variable}__.pkl")

        dff = utils_ml.encode_woe_var_transform(dff, variable, optb )


    return dff