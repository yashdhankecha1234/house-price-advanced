import numpy as np

RAW_COLUMNS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
    "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built",
    "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15",
    "sale_year", "sale_month",
]

def load_data(path="data/kc_house_data.csv"):
    import pandas as pd
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df = df[df["bedrooms"] < 15].reset_index(drop=True)   # 33 bedroom wala outlier hatao
    return df

def add_features(df):
    d = df.copy()
    d["house_age"] = d["sale_year"] - d["yr_built"]
    d["was_renovated"] = (d["yr_renovated"] > 0).astype(int)
    # Downtown Seattle se approx distance (km)
    d["dist_downtown"] = np.sqrt(((d["lat"] - 47.6062) * 111) ** 2 +
                                 ((d["long"] + 122.3321) * 75) ** 2)
    d["total_sqft"] = d["sqft_living"] + d["sqft_basement"]
    d["bath_per_bed"] = d["bathrooms"] / d["bedrooms"].replace(0, 1)
    d["living_vs_neighbors"] = d["sqft_living"] / d["sqft_living15"]
    d["lot_ratio"] = d["sqft_living"] / d["sqft_lot"]
    return d