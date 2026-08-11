import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# CUSTOMER SALES & ANALYTICS DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Customer Sales & Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Theme ----------
INK = "#17202A"
TEAL = "#168F8B"
CORAL = "#E47C5B"
CREAM = "#F5F3EE"
CARD = "#FFFFFF"
MUTED = "#6B7280"
LINE = "#E6E1D9"

st.markdown(f"""
<style>
    .stApp {{
        background: {CREAM};
        color: {INK};
    }}

    [data-testid="stSidebar"] {{
        background: #202A33;
    }}

    [data-testid="stSidebar"] * {{
        color: #F4F4F2 !important;
    }}

    .hero {{
        background: {CARD};
        border: 1px solid {LINE};
        border-left: 7px solid {TEAL};
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 5px 18px rgba(23,32,42,.06);
    }}

    .hero h1 {{
        margin: 0;
        color: {INK};
        font-size: 2.25rem;
        letter-spacing: -1px;
    }}

    .hero p {{
        margin: 7px 0 0;
        color: {MUTED};
        font-size: 1rem;
    }}

    .metric {{
        background: {CARD};
        border: 1px solid {LINE};
        border-radius: 13px;
        padding: 17px 18px;
        box-shadow: 0 3px 12px rgba(23,32,42,.045);
    }}

    .metric-label {{
        color: {MUTED};
        font-size: .82rem;
        text-transform: uppercase;
        letter-spacing: .6px;
    }}

    .metric-value {{
        color: {INK};
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 5px;
    }}

    .section-title {{
        color: {INK};
        font-size: 1.25rem;
        font-weight: 700;
        margin: 20px 0 10px;
    }}

    .insight {{
        background: #E9F5F3;
        border-radius: 12px;
        border: 1px solid #CBE8E5;
        padding: 14px 16px;
        color: {INK};
        margin-top: 12px;
    }}

    .prediction {{
        background: {INK};
        color: white;
        border-radius: 15px;
        padding: 20px;
        margin-top: 12px;
    }}

    .prediction .label {{
        color: #B8C2C8;
        font-size: .8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .prediction .result {{
        color: #FFFFFF;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 5px;
    }}

    .prediction .confidence {{
        color: #BFE4E1;
        margin-top: 5px;
    }}

    .footer {{
        text-align: center;
        color: #8A9095;
        font-size: .8rem;
        padding: 25px 0 10px;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "realistic_e_commerce_sales_data-selected-columns(1).csv")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    uploaded = st.sidebar.file_uploader("Upload sales CSV", type=["csv"])
    if uploaded is None:
        st.warning("Place the project CSV beside this Python file, or upload it from the sidebar.")
        st.stop()
    df = pd.read_csv(uploaded)

required = [
    "Customer ID", "Gender", "Region", "Age", "Product Name",
    "Category", "Unit Price", "Quantity", "Total Price", "Shipping Fee"
]

missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

# Clean source data without changing its business meaning.
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Region"] = df["Region"].fillna("Unknown")

# ============================================================
# CUSTOMER-LEVEL DATASET
# ============================================================

customer_df = (
    df.groupby("Customer ID")
      .agg(
          Gender=("Gender", lambda x: x.mode().iloc[0]),
          Region=("Region", lambda x: x.mode().iloc[0]),
          Age=("Age", "median"),
          Orders=("Customer ID", "size"),
          Avg_Unit_Price=("Unit Price", "mean"),
          Avg_Quantity=("Quantity", "mean"),
          Avg_Shipping=("Shipping Fee", "mean"),
          Total_Spend=("Total Price", "sum"),
          Avg_Order_Value=("Total Price", "mean")
      )
      .reset_index()
)

# Sales tier is created from the actual customer spending distribution.
q1, q2 = customer_df["Total_Spend"].quantile([0.33, 0.67])

def sales_tier(value):
    if value <= q1:
        return "Value"
    elif value <= q2:
        return "Regular"
    return "Premium"

customer_df["Sales_Tier"] = customer_df["Total_Spend"].apply(sales_tier)

# ============================================================
# MODEL
# ============================================================

model_df = customer_df.copy()

cat_cols = ["Gender", "Region"]
model_encoded = pd.get_dummies(
    model_df,
    columns=cat_cols,
    drop_first=True
)

le = LabelEncoder()
model_encoded["Tier_Code"] = le.fit_transform(model_encoded["Sales_Tier"])

features = [
    c for c in model_encoded.columns
    if c not in ["Customer ID", "Sales_Tier", "Tier_Code", "Total_Spend"]
]

X = model_encoded[features]
y = model_encoded["Tier_Code"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=160,
    random_state=42,
    max_depth=8
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

report_df = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
).transpose().round(3)

cm = confusion_matrix(y_test, y_pred)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>Customer Sales & Analytics</h1>
    <p>Interactive overview of customer value, purchasing patterns and sales performance.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("## Dashboard Controls")
st.sidebar.caption("Use filters to explore the sales dataset.")

region_options = ["All"] + sorted(df["Region"].dropna().unique().tolist())
category_options = ["All"] + sorted(df["Category"].unique().tolist())

selected_region = st.sidebar.selectbox("Region", region_options)
selected_category = st.sidebar.selectbox("Category", category_options)

view_df = df.copy()

if selected_region != "All":
    view_df = view_df[view_df["Region"] == selected_region]

if selected_category != "All":
    view_df = view_df[view_df["Category"] == selected_category]

# ============================================================
# KPI STRIP
# ============================================================

total_sales = view_df["Total Price"].sum()
orders = len(view_df)
customers = view_df["Customer ID"].nunique()
avg_order = view_df["Total Price"].mean() if orders else 0

k1, k2, k3, k4 = st.columns(4)

for col, label, value in [
    (k1, "Total Sales", f"₹{total_sales:,.0f}"),
    (k2, "Customers", f"{customers:,}"),
    (k3, "Transactions", f"{orders:,}"),
    (k4, "Avg. Order Value", f"₹{avg_order:,.0f}")
]:
    col.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True
    )

# ============================================================
# ANALYTICS
# ============================================================

st.markdown('<div class="section-title">Sales Overview</div>', unsafe_allow_html=True)

left, right = st.columns([1.25, 1])

category_sales = (
    view_df.groupby("Category", as_index=False)["Total Price"]
    .sum()
    .sort_values("Total Price", ascending=False)
)

fig_category = px.bar(
    category_sales,
    x="Category",
    y="Total Price",
    text_auto=".2s",
    title="Revenue by Category",
    template="simple_white"
)
fig_category.update_traces(marker_color=TEAL)
fig_category.update_layout(
    height=390,
    margin=dict(l=20, r=20, t=55, b=20),
    xaxis_title="",
    yaxis_title="Sales"
)
left.plotly_chart(fig_category, use_container_width=True)

region_sales = (
    view_df.groupby("Region", as_index=False)["Total Price"]
    .sum()
    .sort_values("Total Price", ascending=False)
)

fig_region = px.pie(
    region_sales,
    names="Region",
    values="Total Price",
    hole=.58,
    title="Regional Sales Share",
    template="simple_white"
)
fig_region.update_layout(
    height=390,
    margin=dict(l=10, r=10, t=55, b=10)
)
right.plotly_chart(fig_region, use_container_width=True)

# ---------- Second row ----------
left2, right2 = st.columns(2)

product_sales = (
    view_df.groupby("Product Name", as_index=False)["Total Price"]
    .sum()
    .sort_values("Total Price", ascending=False)
)

fig_product = px.bar(
    product_sales,
    x="Total Price",
    y="Product Name",
    orientation="h",
    title="Top Products by Sales",
    template="simple_white"
)
fig_product.update_traces(marker_color=CORAL)
fig_product.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=55, b=20),
    xaxis_title="Sales",
    yaxis_title=""
)
left2.plotly_chart(fig_product, use_container_width=True)

fig_age = px.scatter(
    view_df,
    x="Age",
    y="Total Price",
    size="Quantity",
    color="Category",
    hover_data=["Customer ID", "Product Name", "Region"],
    title="Customer Age vs Purchase Value",
    template="simple_white"
)
fig_age.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=55, b=20)
)
right2.plotly_chart(fig_age, use_container_width=True)

# ============================================================
# CUSTOMER VALUE SECTION
# ============================================================

st.markdown('<div class="section-title">Customer Value Snapshot</div>', unsafe_allow_html=True)

tier_counts = (
    customer_df["Sales_Tier"]
    .value_counts()
    .reindex(["Premium", "Regular", "Value"])
    .fillna(0)
    .reset_index()
)
tier_counts.columns = ["Sales_Tier", "Customers"]

a, b = st.columns([1, 1.2])

fig_tier = px.bar(
    tier_counts,
    x="Sales_Tier",
    y="Customers",
    text="Customers",
    title="Customer Value Groups",
    template="simple_white"
)
fig_tier.update_traces(marker_color=TEAL)
fig_tier.update_layout(
    height=360,
    margin=dict(l=20, r=20, t=55, b=20),
    xaxis_title="",
    yaxis_title="Customers"
)
a.plotly_chart(fig_tier, use_container_width=True)

feature_importance = (
    pd.Series(model.feature_importances_, index=features)
    .sort_values(ascending=False)
    .head(8)
    .reset_index()
)
feature_importance.columns = ["Feature", "Importance"]

fig_features = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="What Influences Sales Tier Prediction?",
    template="simple_white"
)
fig_features.update_traces(marker_color=CORAL)
fig_features.update_layout(
    height=360,
    margin=dict(l=20, r=20, t=55, b=20),
    yaxis=dict(autorange="reversed")
)
b.plotly_chart(fig_features, use_container_width=True)

premium_share = (
    customer_df["Sales_Tier"].eq("Premium").mean()
    if len(customer_df) else 0
)

st.markdown(
    f"""
    <div class="insight">
        <b>Quick insight:</b> Premium customers represent approximately
        {premium_share:.0%} of the customer base. The dashboard uses
        transaction behaviour such as order frequency, quantity, average
        product price and shipping patterns to estimate customer sales tier.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

with st.expander("Model Performance"):
    m1, m2 = st.columns(2)
    m1.metric("Random Forest Accuracy", f"{accuracy:.2%}")
    m2.metric("Customers Used for Training", f"{len(customer_df):,}")

    cm_fig = px.imshow(
        cm,
        x=list(le.classes_),
        y=list(le.classes_),
        text_auto=True,
        title="Prediction Confusion Matrix",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        template="simple_white"
    )
    cm_fig.update_layout(height=400)
    st.plotly_chart(cm_fig, use_container_width=True)

    st.dataframe(report_df, use_container_width=True)

# ============================================================
# LIVE CUSTOMER PREDICTION
# ============================================================

st.markdown('<div class="section-title">Customer Sales Prediction</div>', unsafe_allow_html=True)

p1, p2 = st.columns([1, 2])

selected_customer = p1.selectbox(
    "Select Customer ID",
    sorted(customer_df["Customer ID"].unique())
)

customer_row = customer_df[
    customer_df["Customer ID"] == selected_customer
].iloc[0]

p1.markdown(
    f"""
    **Orders:** {int(customer_row['Orders'])}  
    **Average Quantity:** {customer_row['Avg_Quantity']:.1f}  
    **Average Order Value:** ₹{customer_row['Avg_Order_Value']:,.0f}
    """
)

if p1.button("Predict Sales Tier", type="primary"):
    raw = customer_row.drop(
        labels=["Customer ID", "Sales_Tier", "Total_Spend"]
    ).to_frame().T

    encoded_input = pd.get_dummies(
        raw,
        columns=["Gender", "Region"],
        drop_first=True
    )

    encoded_input = encoded_input.reindex(
        columns=features,
        fill_value=0
    )

    prediction = model.predict(encoded_input)
    probabilities = model.predict_proba(encoded_input)[0]

    label = le.inverse_transform(prediction)[0]
    confidence = probabilities.max()

    p2.markdown(
        f"""
        <div class="prediction">
            <div class="label">Predicted customer value</div>
            <div class="result">{label} Sales Tier</div>
            <div class="confidence">Model confidence: {confidence:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    prob_df = pd.DataFrame({
        "Sales Tier": le.classes_,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    prob_fig = px.bar(
        prob_df,
        x="Sales Tier",
        y="Probability",
        text_auto=".0%",
        title="Prediction Probability",
        template="simple_white"
    )
    prob_fig.update_traces(marker_color=TEAL)
    prob_fig.update_yaxes(range=[0, 1])
    prob_fig.update_layout(height=330, margin=dict(l=20, r=20, t=55, b=20))
    p2.plotly_chart(prob_fig, use_container_width=True)

# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("View Source Sales Data"):
    st.dataframe(view_df, use_container_width=True, height=330)

st.markdown(
    '<div class="footer">Customer Sales & Analytics • Interactive Business Intelligence Dashboard</div>',
    unsafe_allow_html=True
)
