import os
import math
import xgboost as xgb

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))

def SaveSingleModelInNewVersion(modelName : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  newModelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_1.7.6.json")
  
  booster = xgb.Booster(model_file=modelJSON)
  booster.save_model(newModelJSON)

if __name__ == "__main__":
  print(xgb.__version__)
  SaveSingleModelInNewVersion("abalone")