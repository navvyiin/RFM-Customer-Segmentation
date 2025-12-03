import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import plotly.graph_objects as go

# PAGE CONFIG + THEME
st.set_page_config(
    page_title="Corporate RFM Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY_COLOR = "#0b6e6b"
ACCENT = "#1aa398"
BG = "#f7faf9"
CARD = "#ffffff"
TEXT = "#0b2b2a"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; color: {TEXT}; }}
    .big-font {{ font-size:20px; font-weight:600; color:{PRIMARY_COLOR}; }}
    .kpi {{ background:{CARD}; padding:14px; border-radius:8px;
           box-shadow: 0 4px 18px rgba(0,0,0,0.04); }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
/* Fix invisible metric values */
[data-testid="stMetricValue"] {
    color: #0b2b2a !important;  /* dark text */
}

[data-testid="stMetricDelta"] {
    color: #0b6e6b !important;  /* teal for deltas */
}

/* Fix label colour too */
[data-testid="stMetricLabel"] {
    color: #0b2b2a !important;
}
</style>
""", unsafe_allow_html=True)

# DATA CLEANING + UTILITY FUNCTIONS
def clean_df(df):
    df = df.copy()
    df = df[df["CustomerID"].notna()]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df

def build_rfm(df):
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (ref_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)
    return rfm, ref_date

def scale_rfm(rfm):
    rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_log)
    return scaled, scaler

def compute_k_options(scaled, k_min, k_max):
    inertias, silhouettes = [], []
    K_values = list(range(k_min, k_max+1))
    for k in K_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        inertias.append(km.inertia_)
        try:
            silhouettes.append(silhouette_score(scaled, labels))
        except:
            silhouettes.append(float("nan"))
    return K_values, inertias, silhouettes

def auto_name_segments(rfm):
    summary = rfm.groupby("Cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    })

    z = (summary - summary.mean()) / summary.std(ddof=0)
    labels = {}

    for cid, row in z.iterrows():
        r, f, m = row["Recency"], row["Frequency"], row["Monetary"]

        if r < -0.6 and f > 0.8 and m > 0.8:
            labels[cid] = "Premium Loyalists"
        elif f > 0.5 and m > 0.3:
            labels[cid] = "Active Repeat Buyers"
        elif m > 0.9 and r > 0.6:
            labels[cid] = "High-Value Dormant"
        elif r > 0.7 and f < -0.4:
            labels[cid] = "At-Risk / Churn Likely"
        elif m < -0.6 and f < -0.6:
            labels[cid] = "Bargain-Driven Low Value"
        else:
            labels[cid] = "Emerging Customers"

    return labels

def churn_model_and_scores(rfm_scaled, rfm, recency_threshold_days=180):
    y = (rfm["Recency"] > recency_threshold_days).astype(int)
    X = rfm_scaled

    model = LogisticRegression(max_iter=500)

    try:
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        probs = np.clip(rfm["Recency"] / rfm["Recency"].max(), 0, 1)

    rfm["churn_prob"] = probs
    return model, rfm

def compute_ltv(rfm, df_period_years, profit_margin=0.3, horizon_years=3):
    rfm = rfm.copy()
    rfm["avg_order_value"] = rfm["Monetary"] / rfm["Frequency"].replace(0, np.nan)
    rfm["annual_freq"] = rfm["Frequency"] / max(df_period_years, 1/365)

    rfm["avg_order_value"].fillna(rfm["Monetary"], inplace=True)
    rfm["expected_years"] = (1 - rfm["churn_prob"]) * horizon_years
    rfm["expected_years"] = rfm["expected_years"].clip(lower=0.25, upper=horizon_years)

    rfm["LTV"] = rfm["avg_order_value"] * rfm["annual_freq"] * profit_margin * rfm["expected_years"]
    return rfm

def recommendations_for_segment(name, stats):
    # Accept either naming convention (failsafe)
    ltv = stats.get("LTV_mean", stats.get("avg_LTV"))
    churn = stats.get("churn_prob_mean", stats.get("avg_churn"))

    recs = []

    if ltv is None or churn is None:
        return ["Insufficient data to generate recommendations."]

    if ltv >= 500 and churn < 0.25:
        recs.append("Protect loyalty — VIP experiences, exclusive early access.")
        recs.append("Upsell premium bundles and subscription upgrades.")

    if ltv >= 300 and churn >= 0.25:
        recs.append("Reactivation campaign tailored to past behaviour.")
        recs.append("Investigate possible friction points.")

    if ltv < 300 and churn < 0.4:
        recs.append("Nurturing flows — product education, cross-sell campaigns.")
        recs.append("Encourage habit loops.")

    if churn >= 0.5:
        recs.append("Emergency retention flow — personalised aggressive offers.")
        recs.append("Pause paid acquisition; focus on retention.")

    if not recs:
        recs.append("General lifecycle improvements recommended.")

    return recs

# FILE UPLOAD FIX (FINAL VERSION)
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])

LOCAL_FALLBACK = "data.csv"

def read_csv_flexible(file_obj):
    if isinstance(file_obj, str):
        return pd.read_csv(file_obj, encoding="ISO-8859-1")
    file_obj.seek(0)
    return pd.read_csv(file_obj, encoding="ISO-8859-1")

if uploaded is not None:
    df = read_csv_flexible(uploaded)
else:
    if os.path.exists(LOCAL_FALLBACK):
        df = read_csv_flexible(LOCAL_FALLBACK)
        st.sidebar.info(f"Using fallback: {LOCAL_FALLBACK}")
    else:
        st.sidebar.error("Upload a CSV or place data.csv next to app.py")
        st.stop()

required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

st.success(f"Loaded dataset: {len(df):,} rows")

# SIDEBAR SETTINGS
profit_margin = st.sidebar.slider("Profit Margin", 0.05, 0.7, 0.30, 0.05)
horizon_years = st.sidebar.slider("LTV Horizon (years)", 1, 5, 3)
recency_threshold = st.sidebar.slider("Churn Recency Threshold (days)", 90, 365, 180)
k_min, k_max = st.sidebar.slider("Cluster Search Range", 2, 12, (2, 10))
initial_k = st.sidebar.slider("Selected K", k_min, k_max, 4)
run_kmeans = st.sidebar.button("Recompute")

# DATA PREPARATION
df = clean_df(df)
rfm, ref_date = build_rfm(df)

days_period = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days
years_period = max(days_period / 365, 1/365)

scaled, scaler = scale_rfm(rfm)

K_vals, inertias, silhs = compute_k_options(scaled, k_min, k_max)
selected_k = initial_k

kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=20)
rfm["Cluster"] = kmeans.fit_predict(scaled)

segment_map = auto_name_segments(rfm)
rfm["Segment"] = rfm["Cluster"].map(segment_map)

churn_model, rfm = churn_model_and_scores(scaled, rfm, recency_threshold)
rfm = compute_ltv(rfm, years_period, profit_margin, horizon_years)

cluster_stats = rfm.groupby("Cluster").agg(
    customers=("CustomerID", "count"),
    Recency_mean=("Recency", "mean"),
    Frequency_mean=("Frequency", "mean"),
    Monetary_mean=("Monetary", "mean"),
    churn_prob_mean=("churn_prob", "mean"),
    LTV_mean=("LTV", "mean"),
    Revenue_sum=("Monetary", "sum")
).reset_index()

cluster_stats["Segment"] = cluster_stats["Cluster"].map(segment_map)

# PAGE NAVIGATION
pages = ["Executive Overview", "Cluster Profiles", "LTV & Churn", "AI Recommendations", "Export"]
page = st.sidebar.radio("Navigate", pages)

# PAGE: EXECUTIVE OVERVIEW
if page == "Executive Overview":
    st.markdown("<div class='big-font'>Executive Dashboard</div>", unsafe_allow_html=True)

    total_customers = rfm["CustomerID"].nunique()
    total_revenue = rfm["Monetary"].sum()
    avg_ltv = rfm["LTV"].mean()
    avg_churn = rfm["churn_prob"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><b>Total Customers</b><br><h3>{total_customers:,}</h3></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><b>Total Revenue (£)</b><br><h3>{total_revenue:,.0f}</h3></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><b>Average LTV (£)</b><br><h3>{avg_ltv:.2f}</h3></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><b>Avg Churn Risk</b><br><h3>{avg_churn:.1%}</h3></div>", unsafe_allow_html=True)

    st.markdown("### K Selection Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(x=K_vals, y=inertias, markers=True, title="Elbow Curve"), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(x=K_vals, y=silhs, markers=True, title="Silhouette Scores"), use_container_width=True)

    st.markdown("### Cluster Distribution")
    dist = rfm["Segment"].value_counts().reset_index()
    dist.columns = ["Segment", "Customers"]
    st.plotly_chart(px.bar(dist, x="Segment", y="Customers", color="Segment"), use_container_width=True)

    st.markdown("### Revenue by Segment")
    rev = cluster_stats.groupby("Segment")["Revenue_sum"].sum().reset_index()
    st.plotly_chart(px.pie(rev, names="Segment", values="Revenue_sum"), use_container_width=True)

# PAGE: CLUSTER PROFILES
elif page == "Cluster Profiles":
    st.markdown("<div class='big-font'>Cluster Profiles</div>", unsafe_allow_html=True)

    seg_choice = st.selectbox("Select Segment", sorted(rfm["Segment"].unique()))
    cid = [k for k, v in segment_map.items() if v == seg_choice][0]
    seg_df = rfm[rfm["Cluster"] == cid]

    st.markdown(f"### {seg_choice} — {len(seg_df):,} customers, Revenue £{seg_df['Monetary'].sum():,.2f}")

    # Radar chart
    radar = seg_df[["Recency", "Frequency", "Monetary"]].copy()
    norm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x*0

    r = 1 - norm(radar["Recency"])
    f = norm(radar["Frequency"])
    m = norm(radar["Monetary"])
    vals = [r.mean(), f.mean(), m.mean()]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=["Recency↑", "Frequency", "Monetary", "Recency↑"],
        fill="toself",
        name=seg_choice
    ))
    fig.update_layout(title="Behaviour Radar", polar=dict(radialaxis=dict(range=[0,1])))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Recency", f"{seg_df['Recency'].mean():.1f}")
    c2.metric("Avg Frequency", f"{seg_df['Frequency'].mean():.2f}")
    c3.metric("Avg Monetary (£)", f"{seg_df['Monetary'].mean():.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Avg LTV (£)", f"{seg_df['LTV'].mean():.2f}")
    c5.metric("Avg Churn Risk", f"{seg_df['churn_prob'].mean():.1%}")
    c6.metric("Total Revenue (£)", f"{seg_df['Monetary'].sum():.2f}")

    st.markdown("### Recommended Actions")
    recs = recommendations_for_segment(seg_choice, {
        "LTV_mean": seg_df["LTV"].mean(),
        "churn_prob_mean": seg_df["churn_prob"].mean()
    })
    for r in recs:
        st.success(r)

# PAGE: LTV & CHURN
elif page == "LTV & Churn":
    st.markdown("<div class='big-font'>LTV Forecasting & Churn Analytics</div>", unsafe_allow_html=True)

    st.subheader("Churn Probability Distribution")
    st.plotly_chart(px.histogram(rfm, x="churn_prob", nbins=30), use_container_width=True)

    st.subheader("Customer LTV Distribution")
    st.plotly_chart(px.histogram(rfm, x="LTV", nbins=30), use_container_width=True)

    st.subheader("Segment LTV vs Churn Risk")
    seg = rfm.groupby("Segment").agg({
        "LTV": "mean",
        "churn_prob": "mean",
        "CustomerID": "count"
    }).reset_index()
    fig = px.scatter(seg, x="LTV", y="churn_prob", size="CustomerID", color="Segment")
    st.plotly_chart(fig, use_container_width=True)

# PAGE: AI RECOMMENDATIONS
elif page == "AI Recommendations":
    st.markdown("<div class='big-font'>AI-Driven Marketing Recommendations</div>", unsafe_allow_html=True)

    agg = rfm.groupby("Segment").agg(
        avg_LTV=("LTV", "mean"),
        avg_churn=("churn_prob", "mean"),
        customers=("CustomerID", "count"),
        revenue=("Monetary", "sum")
    ).reset_index().sort_values("avg_LTV", ascending=False)

    st.dataframe(agg.style.format({
        "avg_LTV": "{:.2f}",
        "avg_churn": "{:.2%}",
        "revenue": "{:.2f}"
    }))

    st.markdown("### Recommendations by Segment")

    for _, row in agg.iterrows():
        seg = row["Segment"]
        st.markdown(f"#### {seg}")
        recs = recommendations_for_segment(seg, row)
        for r in recs:
            st.write("- " + r)


# PAGE: EXPORT
elif page == "Export":
    st.markdown("<div class='big-font'>Exports</div>", unsafe_allow_html=True)

    st.download_button(
        "Download Full Segmented Data (CSV)",
        rfm.to_csv(index=False).encode("utf-8"),
        "rfm_segments.csv",
        "text/csv"
    )

    st.download_button(
        "Download Cluster Summary (CSV)",
        cluster_stats.to_csv(index=False).encode("utf-8"),
        "rfm_cluster_summary.csv",
        "text/csv"
    )

# FOOTER
st.markdown("---")
st.markdown(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Corporate RFM Suite")