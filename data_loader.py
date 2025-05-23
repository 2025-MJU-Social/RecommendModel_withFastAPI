import pandas as pd

def load_data():
    contents = pd.read_csv("fixed_contents.csv")
    prices = pd.read_csv("prices.csv")
    return contents, prices
