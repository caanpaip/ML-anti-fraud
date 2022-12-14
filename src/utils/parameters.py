import sys
import platform

## Parameters
if sys.platform=='win32':
    ## path in windows machine
    root = "c:\\Users\\caanp\\OneDrive\\Documents\\repositories\\ML-anti-fraud"
    data_path = "c:\\Users\\caanp\\OneDrive\\Documents\\repositories\\ML-anti-fraud\\data"
    model_path = "c:\\Users\\caanp\\OneDrive\\Documents\\repositories\\ML-anti-fraud\\models"
    features_path = "c:\\Users\\caanp\\OneDrive\\Documents\\repositories\\ML-anti-fraud\\src\\features"

elif sys.platform=='linux':

    if platform.node() == 'pilu':
        ## path in Linux machine in windows
        root = "/home/caanpaip/documents/repositories/ML-anti-fraud"
        data_path = "/home/caanpaip/documents/repositories/ML-anti-fraud/data"
        model_path = "/home/caanpaip/documents/repositories/ML-anti-fraud/models"
        features_path = "/home/caanpaip/documents/repositories/ML-anti-fraud/src/features"

    else:
        ## path in Linux machine
        root = "/home/caanpaip/Documents/GitHub/ML-anti-fraud"
        data_path = "/home/caanpaip/Documents/GitHub/ML-anti-fraud/data"
        model_path = "/home/caanpaip/Documents/GitHub/ML-anti-fraud/models"
        features_path = "/home/caanpaip/Documents/GitHub/ML-anti-fraud/src/features"


cols_rg_model = ['n_boolean_bins_1',
 'p_boolean_bins_1',
 'categoria_produto_bins_1',
 'categoria_produto_bins_2',
 'categoria_produto_bins_3',
 'categoria_produto_bins_4',
 'categoria_produto_bins_5',
 'categoria_produto_bins_6',
 'categoria_produto_bins_7',
 'categoria_produto_bins_8',
 'categoria_produto_bins_9',
 'categoria_produto_bins_10',
 'pais_bins_0',
 'pais_bins_1',
 'a_int_bins_1',
 'b_float_bins_0',
 'b_float_bins_1',
 'b_float_bins_2',
 'b_float_bins_3',
 'b_float_bins_4',
 'b_float_bins_5',
 'c_float_bins_0',
 'c_float_bins_1',
 'd_float_bins_0',
 'd_float_bins_1',
 'd_float_bins_2',
 'd_float_bins_3',
 'd_float_bins_4',
 'd_float_bins_5',
 'd_float_bins_6',
 'e_float_bins_1',
 'e_float_bins_2',
 'f_float_bins_0',
 'f_float_bins_1',
 'f_float_bins_2',
 'f_float_bins_3',
 'f_float_bins_4',
 'f_float_bins_5',
 'f_float_bins_6',
 'f_float_bins_7',
 'f_float_bins_8',
 'f_float_bins_9',
 'h_float_bins_1',
 'h_float_bins_2',
 'h_float_bins_3',
 'h_float_bins_4',
 'h_float_bins_5',
 'k_float_bins_1',
 'k_float_bins_2',
 'k_float_bins_3',
 'k_float_bins_4',
 'l_float_bins_0',
 'l_float_bins_1',
 'l_float_bins_2',
 'l_float_bins_3',
 'l_float_bins_4',
 'l_float_bins_5',
 'l_float_bins_6',
 'l_float_bins_7',
 'l_float_bins_8',
 'l_float_bins_9',
 'l_float_bins_10',
 'l_float_bins_11',
 'm_float_bins_0',
 'm_float_bins_1',
 'm_float_bins_2',
 'm_float_bins_3',
 'm_float_bins_4',
 'm_float_bins_5',
 'm_float_bins_6',
 'm_float_bins_7',
 'm_float_bins_8',
 'm_float_bins_9',
 'm_float_bins_10',
 'm_float_bins_11',
 'monto_bins_1',
 'monto_bins_2',
 'monto_bins_3',
 'monto_bins_4',
 'monto_bins_5',
 'fraude']