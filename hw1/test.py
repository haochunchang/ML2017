import pandas as pd
import feature

test = pd.read_csv("./data/test_X.csv", sep=",", header=None)
test = feature.TestFeature(test)
