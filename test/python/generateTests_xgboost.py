import numpy as np
import pandas as pd
import xgboost as xgb
import os
from xgboost import plot_tree
import matplotlib.pyplot as plt
from glob import glob

model_file_dir = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/test/Random_1Tree"
model_list_file = os.path.join(model_file_dir, "ModelList.txt")
model_list = open(model_list_file, "w")

files = [f for f in glob(model_file_dir + "/*.json")]
for model_file_path in files:
    # input("Waiting ...")
    # model_file_path = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/test/leftheavy_xgb_model.json"
    # model_file_path = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/test/Random_2Tree/TestModel_Size2_1.json"
    # model_file_path = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/testJSON.json"
    print("Processing file : ", model_file_path)
    print(os.path.basename(model_file_path), file=model_list)
    csv_path = model_file_path + ".csv"
    booster = xgb.Booster(model_file=model_file_path)

    numFeatures = booster.num_features()
    numSamples = 1000
    print("Number of features : ", numFeatures)
    x_numpy = -10.0 + 20.0*np.random.rand(numSamples, numFeatures) # np.matrix([[1.1, 0.02, 1.4, 1.3, 0.9]])
    # print(x_numpy)

    # plot_tree(booster)
    # plt.show()

    x = xgb.DMatrix(x_numpy)
    # print("num_col:", x.num_col())
    pred = booster.predict(x)
    pred = np.reshape(pred, (-1, 1))
    # print(pred)
    testArray = np.hstack((x_numpy, pred))
    np.savetxt(csv_path, testArray, delimiter=",")

    # dataframe = pd.DataFrame(testArray)
    # dataframe.to_csv(csv_path)