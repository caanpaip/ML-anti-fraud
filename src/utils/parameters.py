import sys

## Parameters
if sys.platform=='win32':
    ## path in windows machine
    root = "c:\\Users\\caanp\\OneDrive\\Documents\\repositories\\ML-anti-fraud"
    data_path = "c:\\Users\\caanp\\OneDrive\\Documents\\repositories\\ML-anti-fraud\\data"
    model_path = "c:\\Users\\caanp\\OneDrive\\Documents\\repositories\\ML-anti-fraud\\models"
    features_path = "c:\\Users\\caanp\\OneDrive\\Documents\\repositories\\ML-anti-fraud\\src\\features"

elif sys.platform=='linux':
    ## path in Linux machine
    root = "/home/caanpaip/Documents/GitHub/ML-anti-fraud"
    data_path = "/home/caanpaip/Documents/GitHub/ML-anti-fraud/data"
    model_path = "/home/caanpaip/Documents/GitHub/ML-anti-fraud/models"
    features_path = "/home/caanpaip/Documents/GitHub/ML-anti-fraud/src/features"