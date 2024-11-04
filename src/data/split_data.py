from numpy.random import RandomState
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('dataset/viherbqa_official_v2.csv')
    rng = RandomState()

    train = df.sample(frac=0.7, random_state=rng)
    val_test = df.loc[~df.index.isin(train.index)]

    val = val_test.sample(frac=0.9, random_state=rng)
    test = val_test.loc[~val_test.index.isin(val.index)]

    print("Train: ", len(train))
    print("Val: ", len(val))
    print("Test: ", len(test))

    train.to_csv("dataset/train_v2.csv", index=False)
    val.to_csv("dataset/val_v2.csv", index=False)
    test.to_csv("dataset/test_v2.csv", index=False)