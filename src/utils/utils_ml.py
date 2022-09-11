# dev: Carlos Andres Palechor Ipia

import pandas as pd

def missing_zeros(df):
    
    """
    Retorna o número de linhas, zeros, nulos das colunas e a cardinalidade delas
    
        df:= Dataframe de pandas que será usado para o cálculo
    
    """
    ## counting missing or empty values in columns
    nulls_blanks = pd.DataFrame(df.isna().sum() + (df.apply(lambda x: x.astype(str).str.strip())=='').sum()).reset_index(drop = False)
    nulls_blanks.columns = ['Column','T_missing']
    ## Counting distinct values  in columns
    cardinality = pd.DataFrame( df.nunique() ).reset_index(drop = False)
    cardinality.columns = ['Column','Cardinality']
    ## add total rows
    cardinality['T_rows'] = df.shape[0]
    cardinality = cardinality[['Column','T_rows','Cardinality']]
    ## counting zeros in Dataframe  in columns
    zeros =  pd.DataFrame( ( df.apply( lambda x: x.astype(str).str.strip() if x.dtypes == float else x  )=='0'  ).sum() + ( df.apply( lambda x: x.astype(str).str.strip() if x.dtypes != float else x  )==0  ).sum() ).reset_index(drop = False)
    zeros.columns = ['Column','T_zeros']
    ## Extraindo o formato das colunas
    format = pd.DataFrame(df.dtypes).reset_index(drop=False)
    format.columns = ['Column','Type']
    ## columns size in memory
    size_cols = pd.DataFrame( round( (df.memory_usage(index=False, deep=True )/1024)/1024, 2) ).reset_index(drop=False)
    size_cols.columns = ["Column","size MB"]

    df_EA = size_cols.merge( format, on = "Column").merge(cardinality, on='Column' ).merge(nulls_blanks , on='Column').merge(zeros, on='Column')


    df_EA["%_missing"] = 100*(round( df_EA["T_missing"]/df.shape[0],5 ) )
    df_EA["%_zeros"] = 100*(round( df_EA["T_zeros"]/df.shape[0] ,5 ) )


    return df_EA