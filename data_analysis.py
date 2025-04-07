import pandas as pd
import matplotlib.pyplot as plt

import matplotlib; print(matplotlib.__version__)
import PIL; print(PIL.__version__)

# Read CSV file
df = pd.read_csv('historical_data/round-1-island-data-bottle/prices_round_1_day_0.csv', sep=";")

print(df.head())

SQUID_INK = df[df["product"] == "SQUID_INK"]

print(SQUID_INK.columns)
print(SQUID_INK.head())

plt.plot(SQUID_INK["timestamp"], SQUID_INK["bid_price_1"])
plt.show()

pd.set_option('display.max_rows', None)
(SQUID_INK["mid_price"].head(50) - SQUID_INK["mid_price"].head(50).shift(1)).plot()
pd.reset_option('display.max_rows')
plt.show()