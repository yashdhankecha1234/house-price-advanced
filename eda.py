import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import load_data

df = load_data()
print("Shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print(df["price"].describe().round(0))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df["price"], bins=60, color="#2E6F5E"); ax[0].set_title("Price (skewed)")
ax[1].hist(np.log1p(df["price"]), bins=60, color="#E8833A"); ax[1].set_title("log(Price) (balanced)")
plt.tight_layout(); plt.savefig("images/price_distribution.png", dpi=130); plt.close()

num = df.drop(columns=["date"]).select_dtypes("number")
corr = num.corr()["price"].drop("price").sort_values()
plt.figure(figsize=(7, 6))
corr.plot(kind="barh", color="#2E6F5E")
plt.title("Correlation with price"); plt.tight_layout()
plt.savefig("images/correlation.png", dpi=130); plt.close()
print("\nTop correlations:\n", corr.tail(5).round(2))

plt.figure(figsize=(7, 6))
sc = plt.scatter(df["long"], df["lat"], c=np.log1p(df["price"]), s=2, cmap="viridis")
plt.colorbar(sc, label="log(price)"); plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Price by location"); plt.tight_layout()
plt.savefig("images/price_map.png", dpi=130); plt.close()
print("\nSaved 3 images in images/")