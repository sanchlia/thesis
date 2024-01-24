import os 
import pandas as pd 
import numpy as np


PREDICT_FAIR_PC_PATH = os.path.join("para", 'max-ll', 'predict-per-example-proxy-label.csv')
FAIRPC_COLS = ['P(Df) train_x', 'P(Df) test_x', 'P(Df) valid_x', 'P(Df|e) train_x', 'P(Df|e) test_x', 'P(Df|e) valid_x']


def validate_fairPC_assets(base_path):
    # there should be 10 folders in the path
    # each folder should contain a 'predict-per-example-proxy-label.csv'
    # there should be columns 
    # "P(Df) train_x"
    # "P(Df|e) train_x_x
    fols = [os.path.join(base_path, str(x)) for x in range(1,11)]
    for x in fols: 
        if not os.path.exists(x):
            print(f"{x} not present")
        pred_path = os.path.join(x, PREDICT_FAIR_PC_PATH)
        try:
            df = pd.read_csv(pred_path)
            if len([col for col in FAIRPC_COLS if col in df.columns]) != len(FAIRPC_COLS):
                print(f"some columns are missing for {pred_path}")
        except FileNotFoundError as fe:
            print(f"{pred_path} not found")
        



adult_path = "/Users/sauravanchlia/Fair_ML/Projects/thesis/assets/probabilities/FairProbabilities/Adult"
compas_path = "/Users/sauravanchlia/Fair_ML/Projects/thesis/assets/probabilities/FairProbabilities/COMPAS"
income_path = "/Users/sauravanchlia/Fair_ML/Projects/thesis/assets/probabilities/FairProbabilities/income"
synthetic_20_path = "/Users/sauravanchlia/Fair_ML/Projects/thesis/assets/probabilities/FairProbabilities/synthetic_20"


variations = ['bias', 'flip', 'balanced']

# validate_fairPC_assets(adult_path)
# validate_fairPC_assets(compas_path)
# validate_fairPC_assets(income_path)
# validate_fairPC_assets(synthetic_20_path)

for noise in variations:
    for level in ["0.1",'0.3']:
        validate_fairPC_assets(os.path.join(adult_path, noise, level))
    
for noise in variations:
    for level in ["0.1",'0.3']:
        validate_fairPC_assets(os.path.join(compas_path, noise,level))
    
for noise in variations:
    for level in ["0.1",'0.3']:
        validate_fairPC_assets(os.path.join(income_path, noise,level))
    
for noise in variations:
    for level in ["0.1",'0.3']:
        validate_fairPC_assets(os.path.join(synthetic_20_path, noise, level))
    