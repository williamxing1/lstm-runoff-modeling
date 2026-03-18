import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
df = pd.read_csv("basin_metadata.csv")

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.BORDERS)

ax.scatter(df["lon"], df["lat"], s=5, c="r")
plt.title("Basin Locations")
plt.show()