from pathlib import Path
import pandas as pd
from functools import reduce
path = Path("../data/cleaned")

files = [f for f in path.glob("*.csv") if f.stem != "merged_static_features"]
dfs = [pd.read_csv(file, dtype={"gauge_id": str}) for file in files]
df_merged = reduce(lambda left, right: pd.merge(left, right, on="gauge_id", how="outer"), dfs)

# Columns used in Kratzert et al. 2019. gauge_id is needed to match the features with the basin.
# frac_snow_daily -> frac_snow, forest_frac -> frac_forest, carb_rocks_frack -> carbonate_rocks_frac
features = ["gauge_id", "p_mean","pet_mean","aridity","p_seasonality","frac_snow","high_prec_freq","high_prec_dur","low_prec_freq","low_prec_dur","elev_mean","slope_mean","area_gages2","frac_forest","lai_max","lai_diff","gvf_max","gvf_diff","soil_depth_pelletier","soil_depth_statsgo","soil_porosity","soil_conductivity","max_water_content","sand_frac","silt_frac","clay_frac","carbonate_rocks_frac","geol_permeability"]
df_merged = df_merged[features]
df_merged.to_csv("../data/cleaned/merged_static_features.csv", index=False)
print(len(df_merged.columns.tolist()))