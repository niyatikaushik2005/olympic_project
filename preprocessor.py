import pandas as pd

def load(path="data"):
    """Load, clean and merge the Olympic data (Summer games only, 1906 intercalated excluded)."""
    df = pd.read_csv(f"{path}/athlete_events.csv")
    noc = pd.read_csv(f"{path}/noc_regions.csv")
    df = df.drop_duplicates()
    df = df[(df.Season == "Summer") & (df.Year != 1906)]
    df = df.merge(noc[["NOC", "region"]], on="NOC", how="left")
    df["region"] = df["region"].fillna(df["Team"])          # NOCs missing from the mapping
    df["Medal"] = df["Medal"].fillna("None")
    return df.reset_index(drop=True)

def medal_table(df):
    """One medal per (event, NOC): team events are counted once, not once per athlete."""
    m = df[df.Medal != "None"].drop_duplicates(["Year", "Event", "NOC", "Medal"])
    return m
