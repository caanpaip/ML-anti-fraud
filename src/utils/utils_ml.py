# dev: Carlos Andres Palechor Ipia

import pandas as pd
import numpy as np
import os
from unicodedata import normalize
from optbinning import OptimalBinning
import joblib
from scipy.stats import chi2_contingency

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
    format_ = pd.DataFrame(df.dtypes).reset_index(drop=False)
    format_.columns = ['Column','Type']
    ## columns size in memory
    size_cols = pd.DataFrame( round( (df.memory_usage(index=False, deep=True )/1024)/1024, 2) ).reset_index(drop=False)
    size_cols.columns = ["Column","size MB"]
    ## Compute statitics
    df_stats = df.describe(include='all').T.reset_index().drop(['count','unique'], axis=1)
    df_stats.rename({"index":"Column","top":"Mode"}, axis=1, inplace=True )

    ## Join information
    df_EA = size_cols.merge( format_, on = "Column").merge(cardinality, on='Column' ).merge(nulls_blanks , on='Column').merge(zeros, on='Column')

    df_EA["%_missing"] = 100*(round( df_EA["T_missing"]/df.shape[0],5 ) )
    df_EA["%_zeros"] = 100*(round( df_EA["T_zeros"]/df.shape[0] ,5 ) )

    ## Add statitics
    df_EA = df_EA.merge(df_stats,on='Column')
    df_EA["cv"] = df_EA['std']/df_EA['mean']

    ## Fill with blanks
    df_EA.fillna("", inplace=True)


    return df_EA

def no_accents(string):

    """
    Tira acentos e remove os spaços
    """
    
    st=normalize('NFKD', string).encode('ASCII','ignore').decode('ASCII')

    return st.strip().lower()

def metric_evaluation(df, probability, label, quantil):

    """ 
    Return pandas dataframe with performance metrics

        df:= Pandas dataframe with the informations
        probability:= Score column. Need to be float column
        label:= label colums. Need to be integer column
        quantil:= Number of partitions of population to compute the metrics
    
    
    """

    ## Sort by probability column and index created.
    df_sort = df.reset_index().sort_values( by = [probability,'index'] ,ascending=True).drop("index", axis=1).reset_index(drop=True)
    ## Creating the index column
    df_sort = df_sort.reset_index()
    ## Defining the quantiles using the cut pandas function
    df_sort["quantil"] = pd.cut( df_sort['index'], quantil, labels=False)
    ## Aggregate by quantils
    df_group =  df_sort.groupby('quantil').agg( 

                            total = (label,"count"),
                            faixa_prob = ( probability ,lambda x : [min(x),max(x)]),
                            total_um = ( label ,lambda x: sum(x) ),
                            total_zero = (label ,lambda x: sum( (x+1)%2 ) ),

                            )
    ### Creating the false positve annd other by quantil
    df_group["TP"] = df_group['total_um'].sum() - df_group['total_um'].cumsum() + df_group['total_um']
    df_group["FP"] = df_group['total_zero'].sum() - df_group['total_zero'].cumsum() + df_group['total_zero']
    df_group["TN"] = df_group['total_zero'].sum() - df_group["FP"]
    df_group["FN"] = df_group['total_um'].sum() - df_group["TP"]

    ## Compute the % by quantile
    df_group['perc_um'] = 100*df_group['total_um']/df_group['total_um'].sum()
    df_group['perc_zero'] = 100*df_group['total_zero']/df_group['total_zero'].sum()
    ## Compute columns to calcule the KS value
    df_group['csum_um'] = df_group['perc_um'].cumsum()
    df_group['csum_zero'] = df_group['perc_zero'].cumsum()
    ## Ks value
    df_group["ks"] = max(abs(df_group['csum_um'] - df_group['csum_zero']))
    ## Creating the metrics 
    df_group['recall'] = df_group['TP']/(df_group['TP'] + df_group['FN'])
    df_group['precision'] = df_group['TP']/(df_group['TP'] + df_group['FP'])
    df_group['FPR'] = df_group['FP']/(df_group['FP'] + df_group['TP'])
    df_group["BER"] =  0.5*( df_group['FP']/(df_group['TN'] + df_group['FP']) + df_group['FN']/(df_group['TP'] + df_group['FN']) )
    ## Metrics to positive values label=1
    df_group['F1'] = 2*(df_group['precision']*df_group['recall'])/((df_group['precision']+df_group['recall']))
    df_group['F2'] = 5*(df_group['precision']*df_group['recall'])/((4*df_group['precision']+df_group['recall']))

    return df_group


def list_subset_word(List,word):
    """
    função para imprimir elementos (string) de uma lista que contenham a palavra "word"
    """
    list_subset=[]  
    for ls in List:      
        if word in ls:            
            list_subset.append(ls)       
    return list_subset


def list_subset_words(Lista,list_words):

    """
    função para imprimir elementos de uma lista que contehnam uma lista de palavras
    """

    list_subsets = []

    for word in list_words:

        sub_list = list_subset_word(Lista,word)
        list_subsets.extend(sub_list)

    return [i for n, i in enumerate(list_subsets) if i not in list_subsets[:n]]


def exclude_words(ls:list, words:list)->list:

    """
    Return list excluding elements with containd the word in the list words

        ls:= List to exclude elements
        words: List with word to search and exclude elements.


    """

    ls_exclude = ls.copy()

    for word in words:

        ls_exclude = [x for x in ls_exclude if word not in x]

    return ls_exclude

def binning_var_model(df, variable, label, dtype_, monotonic, save_path_pkl = None, verbose = False, optimal_binning = True ):

    """ 
    Retunr object to contruct the binning.

        df:= Dataframe with data
        variable:= Variable in dataframe to compute de metrics.
        dtype:= 'numerical' if the variable is numerical or ordinal variable.
                'categorical' if the variable is nominal variable.
        label:= Target column with values 0 and 1.
        monotonic:= True to compute binning with WoE asceding or descending. Only to numeric variable
                    False to auto.
        save_path_pkl:= Path to save the binning model.
        optimal_binning:= True to compute the WoE using the optimal aprox. 
                           False to compute the WoE using the unique values of variable. Mixing categorias if there is one or more categories without event ou non events. (Only to categorial variables.) 


    Obs: This functions use the optbinning to compute the metrics.
    
        ref: http://gnpalencia.org/optbinning/tutorials/tutorial_binary.html    


    """

    solver = "cp" if dtype_=='numerical' else "mip"
    monotonic_trend = 'auto_asc_desc' if monotonic == True and dtype_=="numerical"  else "auto"
    user_bins =  np.array([[x] for x in df[variable].unique().tolist() ], dtype=object ) if optimal_binning==False and dtype_!="numerical"  else None


    ## using library to compute 
    optb = OptimalBinning(name=variable, dtype=dtype_ , solver=solver, monotonic_trend = monotonic_trend, user_splits = user_bins )
    optb.fit( df[variable].values , df[label])

    if save_path_pkl!=None:

        ## Name to save in the path defined
        name_2_save = os.path.join(save_path_pkl, f"pipeline_binning__{variable}__.pkl")
        ## saving object in pickle
        joblib.dump( optb, name_2_save )

        if verbose == True:

            print("\tModelo salvo em:", name_2_save )
            print(f"\t\tUse joblib.load({name_2_save}) para carregar o modelo")


    return optb



def WOE_IV(df, variable, dtype_, label, monotonic = True, save_path_pkl = None, optimal_binning = True ):

    """
    Retunr WOE, IV and KS of variable in dataset.

        df:= Dataframe with data
        variable:= Variable in dataframe to compute de metrics.
        dtype:= 'numerical' if the variable is numerical or ordinal variable.
                'Categorical' if the variable is nominal variable.
        label:= Target column with values 0 and 1.
        monotonic:= True to compute binning with WoE asceding or descending.
                    False to auto
        save_path_pkl:= Path to save the binning model


    Obs: This functions use the optbinning to compute the metrics.
    
        ref: http://gnpalencia.org/optbinning/tutorials/tutorial_binary.html
    """

    ## using library to compute 
    optb = binning_var_model(df =df , variable = variable , label= label, dtype_ = dtype_ , monotonic = monotonic, save_path_pkl = save_path_pkl, verbose = False, optimal_binning = optimal_binning  )
    ## Creating table 
    binning_table = optb.binning_table
    df_group = binning_table.build(add_totals=False)
    ## Transforming the WoE, If WoE>0 the percentage of event is greather than the Non-event
    df_group["WoE"] = df_group["WoE"].apply(lambda x: -x if x!="" else x)
    ## Extracting the IV value
    df_group["IV_var"] = df_group["IV"].sum()
    df_group["%_Count"] = 100*df_group["Count"]/df_group["Count"].sum()
    ## Selecting the usefull columns
    df_group = df_group[['Bin','%_Count','Count','Non-event','Event','Event rate',"WoE","IV",'IV_var']]
    ## computong the percentage of event or non-event by line
    df_group['%_Event'] = 100*df_group['Event']/df_group['Event'].sum()
    df_group['%_Non-event'] = 100*df_group['Non-event']/df_group['Non-event'].sum()
    ## Computing the KS 
    df_group['KS'] = max(abs(df_group['%_Event'].cumsum() - df_group['%_Non-event'].cumsum()))
    ## Creating column with the variable name.
    df_group["variable"] = variable
    ## Ordering the columns
    df_group = df_group[['variable','Bin','%_Count','Count','Event','Non-event','%_Event','%_Non-event','Event rate',"WoE","IV",'IV_var',"KS"]]


    return df_group


def encode_woe_var_transform(df, variable, opt_model_fitted ):

    """
    Return dataframe with new columns using the model fitted.

        df:= Dataframe with columns to transform
        variable:= Variable in dataframe to compute de metrics.
        opt_model_fitted:= Model obtained of 'binning_var_model' function and load using joblib.load(path_model).
    
    
    """
    df_ = df.copy()

    ## model fitted
    opt = opt_model_fitted
    ## Creating WoE table of model fitted
    df_binning = opt.binning_table.build(add_totals=False)
    ## Extracing WoE of missing category using the last row in dataframe
    woe_missing = -df_binning.loc[df_binning.shape[0]-1:,:].reset_index(drop=True)["WoE"][0]
    # Creating encoding using WoE
    df_[f"{variable}_bins"] = opt.transform(df_[variable], metric="indices", metric_missing= -1)
    df_[f"{variable}_bins"] = df_[f"{variable}_bins"].astype("category")
    df_[f"{variable}_cat"] = opt.transform(df_[variable], metric="bins")
    df_[f"{variable}_WoE"] = opt.transform(df_[variable], metric="woe", metric_missing = woe_missing )
    ## Transforming the WoE, If WoE>0 the percentage of event is greather than the Non-event
    df_[f"{variable}_WoE"] = -df_[f"{variable}_WoE"]

    return df_


def encode_woe_var(df, variable, label, dtype_, monotonic = True, save_path_pkl=None ):

    """ 
    Retunr dataframe with variable encode using WoE.

        df:= Dataframe with data
        variable:= Variable in dataframe to compute de metrics.
        dtype:= 'numerical' if the variable is numerical or ordinal variable.
                'Categorical' if the variable is nominal variable.
        label:= Target column with values 0 and 1.
        monotonic:= True to compute binning with WoE asceding or descending.
                    False to auto
        save_path_pkl:= Path to save the binning model


    Obs: This functions use the optbinning to compute the metrics.

        ref: http://gnpalencia.org/optbinning/tutorials/tutorial_binary.html    


    """
    dff = df.copy()
    ## fit with variable 
    opt = binning_var_model(dff, variable, label, dtype_, monotonic, save_path_pkl )
    ## Creating tranformed columns
    dff = encode_woe_var_transform(dff, variable, opt )

    return dff


def chi_square_test(df, var_cat_1, var_cat_2 ):

    """
        Return dataframe with dependence between two categorical variables using chi-2.

            df:= Dataframe with data
            var_cat_1:= Categorical column 1
            var_cat_2:= Categorical column 2


    https://analyticsindiamag.com/a-beginners-guide-to-chi-square-test-in-python-from-scratch/

    """

    contingency = pd.crosstab( df[var_cat_1],df[var_cat_2], margins=False )

    stat, p, dof, expected = chi2_contingency(observed = contingency)

    alpha = 0.05

    p_value_result = f"p <= {alpha}" if p <= alpha else f"p > {alpha}"
    result = 'Dependent (reject H0)' if p <= alpha else 'Independent (H0 holds true)'


    df_chi_square = pd.DataFrame({"Variable_1":[var_cat_1],"Variable_2":[var_cat_2], "p-value": [p], "p_value_result":[p_value_result], "result": [result]})


    return df_chi_square