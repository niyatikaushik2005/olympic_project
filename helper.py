import plotly.express as px
from preprocessor import medal_table

def sport_cost(r_athletes, base=1.0, per_athlete=0.1):
    """Assumed cost of supporting a sport (budget units). Edit to match real figures."""
    return base + per_athlete * r_athletes

def medals_by_country(df, top=15):
    m = medal_table(df).groupby("region").size().nlargest(top).reset_index(name="medals")
    return px.bar(m, x="region", y="medals", title=f"Top {top} countries by Summer medals")

def participation_trend(df):
    t = df.groupby(["Year", "Sex"]).ID.nunique().reset_index(name="athletes")
    return px.line(t, x="Year", y="athletes", color="Sex", title="Athletes per Games")

def country_sport_medals(df, country):
    m = medal_table(df)
    m = m[m.region == country].groupby(["Sport", "Medal"]).size().reset_index(name="n")
    return px.bar(m, x="Sport", y="n", color="Medal", title=f"{country}: medals by sport",
                  color_discrete_map={"Gold": "#d4af37", "Silver": "#9aa0a6", "Bronze": "#cd7f32"})
