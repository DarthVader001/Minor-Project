import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Customer Support Analytics", layout="wide")

# ------------------- LOAD DATA ---------------------
@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename)
    df["Created_At"] = pd.to_datetime(df["Created_At"], errors="coerce")
    return df

# Make sure your CSV file path is correct
df = load_data(r"Uber_Customer_Support_Tickets_Prepared.csv")

# ------------------- SIDEBAR FILTERS ----------------
st.sidebar.title("Filters")

channel_filter = st.sidebar.multiselect(
    "Channel",
    options=sorted(df["Channel"].dropna().unique()),
    default=sorted(df["Channel"].dropna().unique())
)

issue_filter = st.sidebar.multiselect(
    "Issue Type",
    options=sorted(df["Issue_Type"].dropna().unique()),
    default=sorted(df["Issue_Type"].dropna().unique())
)

min_date = df["Created_At"].min().date()
max_date = df["Created_At"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply filters
mask = (
    df["Channel"].isin(channel_filter) &
    df["Issue_Type"].isin(issue_filter) &
    (df["Created_At"].dt.date >= date_range[0]) &
    (df["Created_At"].dt.date <= date_range[1])
)
df_f = df[mask].copy()

# ------------------- HEADER & KPIs -----------------
st.title("📊 Customer Support Analytics Dashboard")
st.caption("Uber-style support tickets • A/B testing • CSAT • Anomaly insights")
st.write(f"**Filtered tickets:** {len(df_f)} (out of {len(df)})")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Avg CSAT", f"{df_f['CSAT_Score'].mean():.2f}")

with col2:
    st.metric("Avg Resolution Time (min)", f"{df_f['Resolution_Time_Minutes'].mean():.1f}")

with col3:
    st.metric("Tickets per Day (avg)", f"{df_f.groupby(df_f['Created_At'].dt.date).size().mean():.1f}")

with col4:
    happy_rate = df_f["CSAT_Binary"].mean() * 100
    st.metric("Happy Customers (%)", f"{happy_rate:.1f}%")

st.markdown("---")

# ------------------- TABS --------------------------
tab1, tab2, tab3 = st.tabs(["📈 Trends & EDA", "🆚 A/B – Chatbot vs Agent", "⚠️ Anomalies (basic)"])

# ------------------- TAB 1: EDA --------------------
with tab1:
    st.subheader("EDA: Distributions & Trends")

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**CSAT Score Distribution**")
        fig, ax = plt.subplots()
        df_f["CSAT_Score"].plot(kind="hist", bins=5, edgecolor="black", ax=ax)
        ax.set_xlabel("CSAT Score")
        st.pyplot(fig)

    with col_b:
        st.write("**Tickets by Channel**")
        fig, ax = plt.subplots()
        df_f["Channel"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.write("**Daily Ticket Trend**")
    daily_counts = df_f.groupby(df_f["Created_At"].dt.date).size()
    fig, ax = plt.subplots()
    daily_counts.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Tickets")
    st.pyplot(fig)

# ------------------- TAB 2: A/B TEST ----------------
with tab2:
    st.subheader("A/B Testing: Chatbot vs Live Agent (CSAT)")

    ab_df = df_f[df_f["Channel"].isin(["Chatbot", "Live Agent"])].copy()

    if len(ab_df) < 2:
        st.warning("Not enough data for A/B test with current filters.")
    else:
        chatbot = ab_df[ab_df["Channel"]=="Chatbot"]["CSAT_Score"]
        agent   = ab_df[ab_df["Channel"]=="Live Agent"]["CSAT_Score"]

        st.write("**Group Sizes**")
        st.write(f"- Chatbot: {len(chatbot)} tickets")
        st.write(f"- Live Agent: {len(agent)} tickets")

        st.write(f"**Mean CSAT** – Chatbot: {chatbot.mean():.2f}, Agent: {agent.mean():.2f}")

        t_stat, p_val = stats.ttest_ind(chatbot, agent, equal_var=False)
        st.write(f"**Welch t-test** p-value: `{p_val:.4f}`")

        if p_val < 0.05:
            st.success("There IS a statistically significant difference in CSAT between Chatbot and Live Agent (p < 0.05).")
        else:
            st.info("There is NO statistically significant difference in CSAT between Chatbot and Live Agent (p ≥ 0.05).")

        st.write("**CSAT by Channel (Boxplot)**")
        fig, ax = plt.subplots()
        sns.boxplot(x="Channel", y="CSAT_Score", data=ab_df, ax=ax)
        st.pyplot(fig)

# ------------------- TAB 3: ANOMALIES ----------------
with tab3:
    st.subheader("Basic Anomaly View (High Resolution Time)")

    threshold = st.slider("Resolution time threshold (minutes)", min_value=60, max_value=300, value=180, step=10)
    anomalies = df_f[df_f["Resolution_Time_Minutes"] > threshold]

    st.write(f"Tickets with resolution time > {threshold} minutes: **{len(anomalies)}**")

    if not anomalies.empty:
        st.dataframe(
            anomalies[["Ticket_ID","Created_At","Channel","Issue_Type",
                       "Response_Time_Minutes","Resolution_Time_Minutes","CSAT_Score"]].head(20)
        )
    else:
        st.info("No tickets above the selected threshold in current filters.")
