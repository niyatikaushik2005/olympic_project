import pandas as pd
import streamlit as st
from preprocessor import load, medal_table
from feature_engineering import build_dataset, build_future, FEATURES
from synthetic_data import make_synthetic, validate
from model import compare, time_split, conf_matrix
from explainability import explainer_values, summary_plot, waterfall_plot
from optimization import allocate
import helper

st.set_page_config(page_title="Olympic Medal Opportunity", page_icon="🏅", layout="wide")
st.title("🏅 Olympic Medal Opportunity & Resource Planning")
st.caption("Decision-support estimates from historical patterns (Summer Games) - not guarantees of future results.")

@st.cache_data
def get_data():
    df = load()
    return df, build_dataset(df), build_future(df)

@st.cache_resource
def get_models(frac):
    _, data, _ = get_data()
    train, test = time_split(data)
    syn = make_synthetic(train, FEATURES, frac)
    res, fitted = compare(train, test, FEATURES, syn)
    return train, test, syn, res, fitted

df, data, future = get_data()
train, test, syn, res, fitted = get_models(0.5)
xgb = fitted[("XGBoost", "Real")]
future = future.assign(prob=xgb.predict_proba(future[FEATURES])[:, 1])

t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Explore", "🔮 Predictions", "⚖️ Models", "🧪 Synthetic", "🔍 SHAP", "💰 Allocation"])

with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Athlete records", f"{len(df):,}"); c2.metric("Countries", df.region.nunique()); c3.metric("Games", df.Year.nunique())
    a, b = st.columns(2)
    a.plotly_chart(helper.medals_by_country(df))
    b.plotly_chart(helper.participation_trend(df))
    country = st.selectbox("Country", sorted(df.region.dropna().unique()), index=sorted(df.region.dropna().unique()).index("India"), key="eda")
    st.plotly_chart(helper.country_sport_medals(df, country))

with t2:
    st.subheader("Predicted chance of ≥1 medal at the next Games (2020 baseline)")
    st.caption("Dataset ends at 2016, so 2020 is predicted from history up to 2016 for country-sports that played in 2016.")
    c1, c2 = st.columns(2)
    sel = c1.multiselect("Country", sorted(future.region.unique()), default=["India"])
    sports = c2.multiselect("Sport", sorted(future.Sport.unique()))
    v = future.copy()
    if sel: v = v[v.region.isin(sel)]
    if sports: v = v[v.Sport.isin(sports)]
    st.dataframe(v.sort_values("prob", ascending=False)[["region", "Sport", "prob", "h_medals", "r_medals", "r_athletes", "appearances"]]
                 .rename(columns={"region": "Country", "prob": "Medal prob."}), hide_index=True,
                 column_config={"Medal prob.": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.markdown("**What-if:** change a country-sport's inputs and see the prediction move.")
    if len(v):
        row = v.sort_values("prob", ascending=False).iloc[0]
        st.write(f"Scenario base: **{row.region} - {row.Sport}**")
        w1, w2, w3 = st.columns(3)
        ra = w1.slider("Recent athletes", 0, 300, int(row.r_athletes))
        rm = w2.slider("Recent medals", 0, 100, int(row.r_medals))
        hm = w3.slider("Historical medals", 0, 500, int(row.h_medals))
        x = pd.DataFrame([row[FEATURES]]).astype(float)
        x["r_athletes"], x["r_medals"], x["h_medals"] = ra, rm, hm
        x["h_rate"] = hm / max(row.appearances, 1)
        st.metric("Scenario medal probability", f"{xgb.predict_proba(x[FEATURES])[0, 1]:.2f}", f"{xgb.predict_proba(x[FEATURES])[0, 1] - row.prob:+.2f} vs baseline")

with t3:
    st.subheader(f"Time-based evaluation: train < 2008 ({len(train):,} rows), test 2008-2016 ({len(test):,} rows)")
    st.dataframe(res.round(3), hide_index=True)
    name = st.radio("Confusion matrix for", ["XGBoost", "Random Forest"], horizontal=True)
    cm = conf_matrix(fitted[(name, "Real")], test, FEATURES)
    st.dataframe(pd.DataFrame(cm, index=["Actual: no medal", "Actual: medal"], columns=["Pred: no medal", "Pred: medal"]))

with t4:
    st.subheader("Synthetic data (experimental)")
    st.write("Synthetic rows are made by interpolating between pairs of real medal-winning country-sport rows (SMOTE-style), "
             "flagged as synthetic, used **only in training** and never in the test set or any reported historical statistics.")
    ks, cd = validate(train, syn, FEATURES)
    st.write(f"Synthetic rows: **{len(syn):,}** | Mean abs. correlation difference vs real: **{cd:.3f}**")
    st.dataframe(ks.round(4))
    st.caption("A small p-value / large KS statistic means a feature's distribution differs from the real one.")
    st.dataframe(res.pivot(index="Model", columns="Training data", values="ROC-AUC").round(3))

with t5:
    st.subheader("Why does the model predict what it predicts? (XGBoost)")
    sample = test[FEATURES].sample(min(1000, len(test)), random_state=0)
    _, sv = explainer_values(xgb, sample)
    kind = st.radio("Plot", ["beeswarm", "bar"], horizontal=True)
    st.pyplot(summary_plot(sv, kind))
    i = st.slider("Explain individual prediction (sample row)", 0, len(sample) - 1, 0)
    st.pyplot(waterfall_plot(sv[i]))

with t6:
    st.subheader("MILP resource allocation")
    country = st.selectbox("Country", sorted(future.region.unique()), index=sorted(future.region.unique()).index("India"), key="opt")
    c1, c2, c3, c4 = st.columns(4)
    budget = c1.slider("Total budget", 1.0, 100.0, 15.0, 0.5)
    k = c2.slider("Max sports", 1, 15, 5)
    lo = c3.slider("Min funding (x cost)", 0.1, 1.0, 0.5, 0.1)
    hi = c4.slider("Max funding (x cost)", 1.0, 3.0, 1.5, 0.1)
    cand = future[future.region == country][["Sport", "prob", "r_athletes"]].copy()
    cand["cost"] = helper.sport_cost(cand.r_athletes)
    cand = cand.reset_index(drop=True)
    if cand.empty:
        st.info("No candidate sports for this country.")
    else:
        out, val = allocate(cand, budget, k, lo, hi)
        st.metric("Objective (sum of prob x funding/cost)", f"{val:.2f}")
        if out.empty: st.warning("Budget too small to fund any sport under these limits.")
        else:
            st.dataframe(out[["Sport", "prob", "cost", "allocation"]].round(2), hide_index=True)
            st.bar_chart(out.set_index("Sport").allocation)
        st.caption("Costs are assumed (1 + 0.1 x recent athletes). Results are a scenario under these constraints, not a funding verdict.")
