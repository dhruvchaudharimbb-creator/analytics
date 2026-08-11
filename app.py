import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# CUSTOMER SALES & ANALYTICS DASHBOARD
# Streamlit Cloud friendly - no Plotly dependency required
# ============================================================

st.set_page_config(
    page_title="Customer Sales & Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- STYLE ----------------------
st.markdown("""
<style>
    .stApp {
        background: #F4F2ED;
    }

    [data-testid="stSidebar"] {
        background: #202A33;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        background: white;
        padding: 24px 28px;
        border-radius: 16px;
        border: 1px solid #DDD9D0;
        border-left: 7px solid #158F8B;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 750;
        color: #17202A;
        margin: 0;
    }

    .hero-sub {
        color: #6B7280;
        margin-top: 6px;
        font-size: 15px;
    }

    .metric-card {
        background: white;
        border: 1px solid #DDD9D0;
        border-radius: 13px;
        padding: 16px;
        min-height: 105px;
    }

    .metric-label {
        color: #73777C;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .metric-value {
        color: #17202A;
        font-size: 25px;
        font-weight: 750;
        margin-top: 7px;
    }

    .prediction-card {
        background: #17202A;
        color: white;
        padding: 22px;
        border-radius: 15px;
        border-left: 6px solid #158F8B;
    }

    .prediction-label {
        color: #B8C1C7;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-result {
        font-size: 28px;
        font-weight: 750;
        margin-top: 5px;
    }

    .prediction-confidence {
        color: #B9E5E2;
        margin-top: 5px;
    }

    .insight {
        background: #E5F3F1;
        border: 1px solid #C5E5E1;
        padding: 14px 17px;
        border-radius: 12px;
        color: #17202A;
        margin: 8px 0 18px;
    }

    .footer {
        text-align: center;
        color: #777D82;
        padding: 25px 0 10px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD CSV
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_NAME = "realistic_e_commerce_sales_data-selected-columns(1).csv"
CSV_PATH = os.path.join(BASE_DIR, CSV_NAME)

@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    # Fallback for Streamlit Cloud if the CSV is not found beside app.py.
    return None

df = load_data()

if df is None:
    st.error(
        "CSV file not found. Upload the CSV using the sidebar, "
        "or place it in the same GitHub folder as app.py."
    )
    uploaded_file = st.sidebar.file_uploader(
        "Upload the e-commerce CSV",
        type=["csv"]
    )

    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file)


# ============================================================
# CHECK DATASET
# ============================================================

REQUIRED_COLUMNS = [
    "Customer ID",
    "Gender",
    "Region",
    "Age",
    "Product Name",
    "Category",
    "Unit Price",
    "Quantity",
    "Total Price",
    "Shipping Fee"
]

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    st.error(
        "The uploaded CSV is missing: "
        + ", ".join(missing_columns)
    )
    st.stop()

# Numeric cleanup
numeric_columns = [
    "Age",
    "Unit Price",
    "Quantity",
    "Total Price",
    "Shipping Fee"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Region"] = df["Region"].fillna("Unknown")

# Remove unusable rows only if essential fields are missing.
df = df.dropna(
    subset=["Customer ID", "Total Price", "Quantity"]
).copy()

if df.empty:
    st.error("The CSV does not contain usable sales records.")
    st.stop()


# ============================================================
# CUSTOMER-LEVEL ANALYTICS
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

# Customer sales tiers are derived from the actual customer spending.
# Total_Spend is deliberately NOT used as a model feature.
try:
    customer_df["Sales_Tier"] = pd.qcut(
        customer_df["Total_Spend"],
        q=3,
        labels=["Value", "Regular", "Premium"],
        duplicates="drop"
    ).astype(str)
except ValueError:
    customer_df["Sales_Tier"] = "Regular"

# Safety check: ensure there are at least two classes.
if customer_df["Sales_Tier"].nunique() < 2:
    customer_df["Sales_Tier"] = pd.cut(
        customer_df["Total_Spend"],
        bins=3,
        labels=["Value", "Regular", "Premium"],
        include_lowest=True
    ).astype(str)

customer_df["Sales_Tier"] = customer_df["Sales_Tier"].replace(
    {"nan": "Regular"}
)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

model_df = customer_df.copy()

categorical_columns = ["Gender", "Region"]

encoded_df = pd.get_dummies(
    model_df,
    columns=categorical_columns,
    drop_first=True
)

label_encoder = LabelEncoder()
encoded_df["Tier_Code"] = label_encoder.fit_transform(
    encoded_df["Sales_Tier"]
)

excluded_columns = [
    "Customer ID",
    "Sales_Tier",
    "Tier_Code",
    "Total_Spend"
]

feature_columns = [
    col for col in encoded_df.columns
    if col not in excluded_columns
]

X = encoded_df[feature_columns]
y = encoded_df["Tier_Code"]

# Fill any remaining numerical gaps safely.
X = X.fillna(0)

if y.nunique() >= 2 and len(customer_df) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose().round(3)

    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=range(len(label_encoder.classes_))
    )

else:
    model = None
    accuracy = 0
    report_df = pd.DataFrame()
    confusion = None


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">Customer Sales & Analytics</div>
    <div class="hero-sub">
        A practical view of customer purchasing behaviour, sales performance
        and customer value prediction.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Dashboard Controls")
st.sidebar.caption("Filter the sales records below.")

region_values = sorted(
    df["Region"].dropna().astype(str).unique().tolist()
)

category_values = sorted(
    df["Category"].dropna().astype(str).unique().tolist()
)

selected_region = st.sidebar.selectbox(
    "Region",
    ["All"] + region_values
)

selected_category = st.sidebar.selectbox(
    "Category",
    ["All"] + category_values
)

view_df = df.copy()

if selected_region != "All":
    view_df = view_df[
        view_df["Region"].astype(str) == selected_region
    ]

if selected_category != "All":
    view_df = view_df[
        view_df["Category"].astype(str) == selected_category
    ]


# ============================================================
# KPI CARDS
# ============================================================

total_sales = view_df["Total Price"].sum()
transaction_count = len(view_df)
customer_count = view_df["Customer ID"].nunique()

if transaction_count > 0:
    average_order = view_df["Total Price"].mean()
else:
    average_order = 0

k1, k2, k3, k4 = st.columns(4)

metrics = [
    (k1, "Total Sales", f"₹{total_sales:,.0f}"),
    (k2, "Customers", f"{customer_count:,}"),
    (k3, "Transactions", f"{transaction_count:,}"),
    (k4, "Average Order", f"₹{average_order:,.0f}")
]

for col, label, value in metrics:
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SALES OVERVIEW
# ============================================================

st.subheader("Sales Overview")

left, right = st.columns(2)

# Category sales
category_sales = (
    view_df.groupby("Category")["Total Price"]
    .sum()
    .sort_values(ascending=False)
)

with left:
    st.markdown("**Sales by Category**")
    if not category_sales.empty:
        st.bar_chart(category_sales, use_container_width=True)
    else:
        st.info("No category data for the selected filters.")

# Region sales
region_sales = (
    view_df.groupby("Region")["Total Price"]
    .sum()
    .sort_values(ascending=False)
)

with right:
    st.markdown("**Sales by Region**")
    if not region_sales.empty:
        st.bar_chart(region_sales, use_container_width=True)
    else:
        st.info("No region data for the selected filters.")


# ============================================================
# SECOND ANALYTICS ROW
# ============================================================

left, right = st.columns(2)

# Product sales
product_sales = (
    view_df.groupby("Product Name")["Total Price"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

with left:
    st.markdown("**Top 10 Products by Sales**")
    if not product_sales.empty:
        st.bar_chart(product_sales, use_container_width=True)
    else:
        st.info("No product data available.")

# Age analysis
age_sales = (
    view_df.groupby("Age")["Total Price"]
    .sum()
    .sort_index()
)

with right:
    st.markdown("**Sales by Customer Age**")
    if not age_sales.empty:
        st.line_chart(age_sales, use_container_width=True)
    else:
        st.info("No age data available.")


# ============================================================
# CUSTOMER VALUE
# ============================================================

st.subheader("Customer Value Analysis")

tier_order = ["Value", "Regular", "Premium"]

tier_counts = (
    customer_df["Sales_Tier"]
    .value_counts()
    .reindex(tier_order)
    .fillna(0)
    .astype(int)
)

left, right = st.columns(2)

with left:
    st.markdown("**Customers by Sales Tier**")
    st.bar_chart(tier_counts, use_container_width=True)

with right:
    st.markdown("**Customer Tier Summary**")

    tier_summary = (
        customer_df.groupby("Sales_Tier")
        .agg(
            Customers=("Customer ID", "count"),
            Total_Sales=("Total_Spend", "sum"),
            Avg_Spend=("Total_Spend", "mean")
        )
        .reindex(tier_order)
        .fillna(0)
    )

    tier_summary["Total_Sales"] = tier_summary["Total_Sales"].round(0)
    tier_summary["Avg_Spend"] = tier_summary["Avg_Spend"].round(0)

    st.dataframe(
        tier_summary,
        use_container_width=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if model is not None:
    st.subheader("Prediction Drivers")

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    ).head(8)

    importance_df = importance_df.set_index("Feature")

    st.bar_chart(
        importance_df["Importance"],
        use_container_width=True
    )

    premium_percentage = (
        customer_df["Sales_Tier"].eq("Premium").mean() * 100
    )

    st.markdown(
        f"""
        <div class="insight">
            <b>Business insight:</b>
            Premium customers represent approximately
            {premium_percentage:.1f}% of the customer base.
            The prediction model learns from customer behaviour such as
            order count, average quantity, average product price,
            shipping behaviour, age and region.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

with st.expander("View Model Performance"):

    if model is None:
        st.warning("Model performance cannot be calculated for this dataset.")
    else:
        a, b = st.columns(2)

        a.metric("Random Forest Accuracy", f"{accuracy:.2%}")
        b.metric("Customer Records", f"{len(customer_df):,}")

        st.markdown("**Confusion Matrix**")

        confusion_df = pd.DataFrame(
            confusion,
            index=[
                f"Actual: {x}"
                for x in label_encoder.classes_
            ],
            columns=[
                f"Predicted: {x}"
                for x in label_encoder.classes_
            ]
        )

        st.dataframe(
            confusion_df,
            use_container_width=True
        )

        st.markdown("**Classification Report**")
        st.dataframe(
            report_df,
            use_container_width=True
        )


# ============================================================
# LIVE CUSTOMER PREDICTION
# ============================================================

st.subheader("Live Customer Prediction")

if model is None:
    st.warning("Prediction is unavailable because the model could not be trained.")
else:
    selected_customer = st.selectbox(
        "Select Customer ID",
        sorted(customer_df["Customer ID"].astype(str).unique())
    )

    selected_row = customer_df[
        customer_df["Customer ID"].astype(str) == selected_customer
    ].iloc[0]

    p1, p2 = st.columns([1, 1.6])

    with p1:
        st.markdown("**Customer Profile**")
        st.write(f"**Region:** {selected_row['Region']}")
        st.write(f"**Age:** {selected_row['Age']:.0f}")
        st.write(f"**Orders:** {int(selected_row['Orders'])}")
        st.write(
            f"**Average Quantity:** "
            f"{selected_row['Avg_Quantity']:.1f}"
        )
        st.write(
            f"**Average Order Value:** "
            f"₹{selected_row['Avg_Order_Value']:,.0f}"
        )

        predict_button = st.button(
            "Predict Customer Tier",
            type="primary"
        )

    if predict_button:
        input_row = selected_row.drop(
            labels=[
                "Customer ID",
                "Sales_Tier",
                "Total_Spend"
            ]
        ).to_frame().T

        input_encoded = pd.get_dummies(
            input_row,
            columns=["Gender", "Region"],
            drop_first=True
        )

        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        input_encoded = input_encoded.fillna(0)

        prediction = model.predict(input_encoded)[0]
        probabilities = model.predict_proba(input_encoded)[0]

        predicted_label = label_encoder.inverse_transform(
            [prediction]
        )[0]

        confidence = probabilities.max()

        with p2:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        Predicted Customer Value
                    </div>
                    <div class="prediction-result">
                        {predicted_label}
                    </div>
                    <div class="prediction-confidence">
                        Confidence: {confidence:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            probability_df = pd.DataFrame({
                "Sales Tier": label_encoder.classes_,
                "Probability": probabilities
            }).set_index("Sales Tier")

            st.markdown("**Prediction Probability**")
            st.bar_chart(
                probability_df["Probability"],
                use_container_width=True
            )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("View Source Sales Data"):
    st.dataframe(
        view_df,
        use_container_width=True,
        height=350
    )

st.markdown(
    '<div class="footer">Customer Sales & Analytics Dashboard</div>',
    unsafe_allow_html=True
)
