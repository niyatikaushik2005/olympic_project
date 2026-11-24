import pandas as pd
from preprocessor import medal_table

KEYS = ["region", "Sport"]
FEATURES = ["appearances", "h_athletes", "h_events", "h_gold", "h_silver", "h_bronze",
            "h_medals", "h_rate", "r_athletes", "r_medals", "r_appear", "years_since_last"]

def build_panel(df):
    """Athlete-level records -> one row per (country, sport, Olympic year)."""
    base = df.groupby(KEYS + ["Year"]).agg(athletes=("ID", "nunique"), events=("Event", "nunique"))
    m = medal_table(df)
    med = (m.groupby(["region", "Sport", "Year", "Medal"]).size().unstack(fill_value=0)
             .reindex(columns=["Gold", "Silver", "Bronze"], fill_value=0))
    p = base.join(med, how="left").fillna(0).reset_index()
    p.columns = [c.lower() if c in ("Gold", "Silver", "Bronze") else c for c in p.columns]
    p["medals"] = p.gold + p.silver + p.bronze
    p["target"] = (p.medals > 0).astype(int)
    return p

def add_history(panel, window=8):
    """Features use ONLY editions strictly before the row's year (no leakage)."""
    h = panel.merge(panel, on=KEYS, suffixes=("", "_h"))
    h = h[h.Year_h < h.Year]
    allh = h.groupby(KEYS + ["Year"]).agg(
        appearances=("Year_h", "nunique"), h_athletes=("athletes_h", "sum"),
        h_events=("events_h", "mean"), h_gold=("gold_h", "sum"), h_silver=("silver_h", "sum"),
        h_bronze=("bronze_h", "sum"), h_medals=("medals_h", "sum"), last=("Year_h", "max"))
    rec = h[h.Year_h >= h.Year - window].groupby(KEYS + ["Year"]).agg(
        r_athletes=("athletes_h", "sum"), r_medals=("medals_h", "sum"), r_appear=("Year_h", "nunique"))
    out = panel.merge(allh.reset_index(), on=KEYS + ["Year"], how="left") \
               .merge(rec.reset_index(), on=KEYS + ["Year"], how="left")
    out[["appearances", "h_athletes", "h_events", "h_gold", "h_silver", "h_bronze", "h_medals",
         "r_athletes", "r_medals", "r_appear"]] = out[["appearances", "h_athletes", "h_events", "h_gold",
         "h_silver", "h_bronze", "h_medals", "r_athletes", "r_medals", "r_appear"]].fillna(0)
    out["h_rate"] = out.h_medals / out.appearances.replace(0, 1)
    out["years_since_last"] = (out.Year - out["last"]).fillna(40)
    return out.drop(columns="last")

def build_dataset(df):
    """Labelled rows: every country-sport with at least one earlier appearance."""
    d = add_history(build_panel(df))
    return d[d.appearances > 0].reset_index(drop=True)

def build_future(df, year=2020):
    """Feature rows for a future edition: country-sports that played in the latest edition."""
    p = build_panel(df)
    last = p[p.Year == p.Year.max()][KEYS].copy()
    fut = last.assign(Year=year, athletes=0, events=0, gold=0, silver=0, bronze=0, medals=0, target=0)
    d = add_history(pd.concat([p, fut], ignore_index=True))
    return d[d.Year == year].reset_index(drop=True)
