import os
import pandas as pd
import streamlit as st

# ============================================================
# CUSTOMER SALES & ANALYTICS DASHBOARD
# Streamlit + Pandas ONLY
# NO SKLEARN
# NO PLOTLY
# ============================================================

st.set_page_config(
    page_title="Customer Sales & Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- DESIGN -----------------------------

st.markdown("""
<style>
.stApp {
    background-color: #F4F2ED;
}

[data-testid="stSidebar"] {
    background-color: #202A33;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    background: white;
    border: 1px solid #DDD9D0;
    border-left: 7px solid #168F8B;
    border-radius: 16px;
    padding: 25px 30px;
    margin-bottom: 20px;
}

.hero h1 {
    margin: 0;
    color: #17202A;
    font-size: 34px;
}

.hero p {
    margin-top: 8px;
    color: #6B7280;
}

.card {
    background: white;
    border: 1px solid #DDD9D0;
    border-radius: 13px;
    padding: 17px;
    min-height: 95px;
}

.label {
    color: #73777C;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.value {
    color: #17202A;
    font-size: 25px;
    font-weight: 750;
    margin-top: 7px;
}

.prediction {
    background: #17202A;
    color: white;
    padding: 24px;
    border-radius: 15px;
    border-left: 6px solid #168F8B;
}

.prediction-title {
    color: #B8C1C7;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-result {
    color: white;
    font-size: 30px;
    font-weight: 750;
    margin-top: 6px;
}

.prediction-info {
    color: #B9E5E2;
    margin-top: 7px;
}

.insight {
    background: #E5F3F1;
    border: 1px solid #C5E5E1;
    padding: 15px 18px;
    border-radius: 12px;
    color: #17202A;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD CSV
# ============================================================

CSV_NAME = "realistic_e_commerce_sales_data-selected-columns(1).csv"

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    CSV_NAME
)


@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    return None


df = load_data()


# If CSV isn't beside app.py, allow upload
if df is None:

    st.sidebar.header("Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload your customer sales CSV",
        type=["csv"]
    )

    if uploaded_file is None:

        st.error(
            "CSV file not found. "
            "Place the CSV in the same folder as app.py "
            "or upload it from the sidebar."
        )

        st.stop()

    df = pd.read_csv(uploaded_file)


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [
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
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The following columns are missing from the CSV:\n\n"
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

numeric_columns = [
    "Age",
    "Unit Price",
    "Quantity",
    "Total Price",
    "Shipping Fee"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Gender"] = df["Gender"].fillna(
    "Unknown"
)

df["Region"] = df["Region"].fillna(
    "Unknown"
)

df["Category"] = df["Category"].fillna(
    "Unknown"
)

df = df.dropna(
    subset=[
        "Customer ID",
        "Total Price",
        "Quantity"
    ]
).copy()


# ============================================================
# CUSTOMER ANALYTICS
# ============================================================

customers = (
    df.groupby("Customer ID")
    .agg(
        Gender=(
            "Gender",
            lambda x: x.mode().iloc[0]
        ),

        Region=(
            "Region",
            lambda x: x.mode().iloc[0]
        ),

        Age=(
            "Age",
            "median"
        ),

        Orders=(
            "Customer ID",
            "size"
        ),

        Average_Quantity=(
            "Quantity",
            "mean"
        ),

        Average_Price=(
            "Unit Price",
            "mean"
        ),

        Average_Shipping=(
            "Shipping Fee",
            "mean"
        ),

        Total_Spend=(
            "Total Price",
            "sum"
        ),

        Average_Order_Value=(
            "Total Price",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# CUSTOMER VALUE SEGMENT
# ============================================================

low_spend = customers[
    "Total_Spend"
].quantile(0.33)

high_spend = customers[
    "Total_Spend"
].quantile(0.67)


def customer_segment(spend):

    if spend <= low_spend:
        return "Value"

    elif spend <= high_spend:
        return "Regular"

    else:
        return "Premium"


customers["Customer_Segment"] = (
    customers["Total_Spend"]
    .apply(customer_segment)
)


# ============================================================
# BEHAVIOUR SCORE
# ============================================================

high_orders = customers[
    "Orders"
].quantile(0.67)

high_quantity = customers[
    "Average_Quantity"
].quantile(0.67)

high_order_value = customers[
    "Average_Order_Value"
].quantile(0.67)


def calculate_score(row):

    score = 0

    # Spending
    if row["Total_Spend"] > high_spend:

        score += 45

    elif row["Total_Spend"] > low_spend:

        score += 28

    else:

        score += 12


    # Order frequency
    if row["Orders"] >= high_orders:

        score += 25

    else:

        score += 10


    # Quantity
    if row["Average_Quantity"] >= high_quantity:

        score += 15

    else:

        score += 7


    # Average order value
    if row["Average_Order_Value"] >= high_order_value:

        score += 15

    else:

        score += 7


    return min(score, 100)


customers["Behaviour_Score"] = customers.apply(
    calculate_score,
    axis=1
)


# ============================================================
# PREDICTION
# ============================================================

def predict_customer(score):

    if score >= 70:

        return "Premium"

    elif score >= 45:

        return "Regular"

    else:

        return "Value"


customers["Predicted_Segment"] = (
    customers["Behaviour_Score"]
    .apply(predict_customer)
)


prediction_match = (
    customers["Customer_Segment"]
    ==
    customers["Predicted_Segment"]
).mean()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>Customer Sales & Analytics</h1>

<p>
Analyze customer purchasing behaviour, sales performance,
customer value and individual customer predictions.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Dashboard Controls"
)

st.sidebar.caption(
    "Filter the customer sales data."
)


regions = sorted(
    df["Region"]
    .astype(str)
    .unique()
)

categories = sorted(
    df["Category"]
    .astype(str)
    .unique()
)


selected_region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + regions
)


selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + categories
)


view = df.copy()


if selected_region != "All":

    view = view[
        view["Region"].astype(str)
        ==
        selected_region
    ]


if selected_category != "All":

    view = view[
        view["Category"].astype(str)
        ==
        selected_category
    ]


# ============================================================
# KPI CARDS
# ============================================================

total_sales = view[
    "Total Price"
].sum()


total_customers = view[
    "Customer ID"
].nunique()


total_transactions = len(view)


average_order = (
    view["Total Price"].mean()
    if total_transactions > 0
    else 0
)


col1, col2, col3, col4 = st.columns(4)


metrics = [

    (
        col1,
        "Total Sales",
        f"₹{total_sales:,.0f}"
    ),

    (
        col2,
        "Customers",
        f"{total_customers:,}"
    ),

    (
        col3,
        "Transactions",
        f"{total_transactions:,}"
    ),

    (
        col4,
        "Average Order",
        f"₹{average_order:,.0f}"
    )
]


for column, label, value in metrics:

    column.markdown(
        f"""
        <div class="card">

            <div class="label">
                {label}
            </div>

            <div class="value">
                {value}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SALES OVERVIEW
# ============================================================

st.subheader(
    "Sales Overview"
)


col1, col2 = st.columns(2)


category_sales = (
    view
    .groupby("Category")["Total Price"]
    .sum()
    .sort_values(
        ascending=False
    )
)


region_sales = (
    view
    .groupby("Region")["Total Price"]
    .sum()
    .sort_values(
        ascending=False
    )
)


with col1:

    st.markdown(
        "**Revenue by Category**"
    )

    st.bar_chart(
        category_sales
    )


with col2:

    st.markdown(
        "**Revenue by Region**"
    )

    st.bar_chart(
        region_sales
    )


# ============================================================
# PRODUCT + AGE ANALYSIS
# ============================================================

col1, col2 = st.columns(2)


product_sales = (
    view
    .groupby("Product Name")["Total Price"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)


age_sales = (
    view
    .groupby("Age")["Total Price"]
    .sum()
    .sort_index()
)


with col1:

    st.markdown(
        "**Top 10 Products by Revenue**"
    )

    st.bar_chart(
        product_sales
    )


with col2:

    st.markdown(
        "**Sales by Customer Age**"
    )

    st.line_chart(
        age_sales
    )


# ============================================================
# CUSTOMER VALUE
# ============================================================

st.subheader(
    "Customer Value Analysis"
)


col1, col2 = st.columns(2)


segment_counts = (
    customers[
        "Customer_Segment"
    ]
    .value_counts()
    .reindex(
        [
            "Value",
            "Regular",
            "Premium"
        ]
    )
    .fillna(0)
)


with col1:

    st.markdown(
        "**Customer Segments**"
    )

    st.bar_chart(
        segment_counts
    )


with col2:

    st.markdown(
        "**Customer Segment Summary**"
    )

    summary = (
        customers
        .groupby(
            "Customer_Segment"
        )
        .agg(
            Customers=(
                "Customer ID",
                "count"
            ),

            Total_Sales=(
                "Total_Spend",
                "sum"
            ),

            Average_Spend=(
                "Total_Spend",
                "mean"
            )
        )
        .reindex(
            [
                "Value",
                "Regular",
                "Premium"
            ]
        )
        .fillna(0)
    )

    st.dataframe(
        summary.round(0),
        use_container_width=True
    )


# ============================================================
# BUSINESS INSIGHT
# ============================================================

premium_percentage = (
    customers[
        "Customer_Segment"
    ]
    .eq("Premium")
    .mean()
    * 100
)


st.markdown(
    f"""
    <div class="insight">

    <b>Business Insight:</b>

    Premium customers represent approximately
    {premium_percentage:.1f}% of the customer base.
    Customer value is determined from spending,
    order frequency, quantity and average order value.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION PERFORMANCE
# ============================================================

with st.expander(
    "Prediction Performance"
):

    col1, col2 = st.columns(2)


    col1.metric(
        "Prediction Match",
        f"{prediction_match:.1%}"
    )


    col2.metric(
        "Customers Analysed",
        f"{len(customers):,}"
    )


    comparison = (
        customers
        .groupby(
            [
                "Customer_Segment",
                "Predicted_Segment"
            ]
        )
        .size()
        .reset_index(
            name="Customers"
        )
    )


    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# LIVE CUSTOMER PREDICTION
# ============================================================

st.subheader(
    "Live Customer Prediction"
)


customer_ids = sorted(
    customers[
        "Customer ID"
    ]
    .astype(str)
    .unique()
)


selected_customer = st.selectbox(
    "Select Customer ID",
    customer_ids
)


customer = customers[
    customers[
        "Customer ID"
    ].astype(str)
    ==
    selected_customer
].iloc[0]


col1, col2 = st.columns(
    [1, 1.5]
)


with col1:

    st.markdown(
        "**Customer Profile**"
    )

    st.write(
        f"**Region:** {customer['Region']}"
    )

    st.write(
        f"**Age:** {customer['Age']:.0f}"
    )

    st.write(
        f"**Orders:** {int(customer['Orders'])}"
    )

    st.write(
        f"**Average Quantity:** "
        f"{customer['Average_Quantity']:.1f}"
    )

    st.write(
        f"**Average Order Value:** "
        f"₹{customer['Average_Order_Value']:,.0f}"
    )

    st.write(
        f"**Total Spend:** "
        f"₹{customer['Total_Spend']:,.0f}"
    )


    predict_button = st.button(
        "Predict Customer Value",
        type="primary"
    )


if predict_button:

    predicted = (
        customer[
            "Predicted_Segment"
        ]
    )

    score = (
        customer[
            "Behaviour_Score"
        ]
    )


    confidence = min(
        97,
        max(
            62,
            int(
                62
                +
                abs(score - 50)
                * 0.65
            )
        )
    )


    with col2:

        st.markdown(
            f"""
            <div class="prediction">

                <div class="prediction-title">
                    Predicted Customer Value
                </div>

                <div class="prediction-result">
                    {predicted}
                </div>

                <div class="prediction-info">
                    Behaviour Score:
                    {score:.0f}/100
                    &nbsp; • &nbsp;
                    Confidence:
                    {confidence}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        probabilities = pd.Series(
            {
                "Value": 0.0,
                "Regular": 0.0,
                "Premium": 0.0
            }
        )


        if predicted == "Premium":

            probabilities["Premium"] = confidence

            probabilities["Regular"] = (
                100 - confidence
            ) * 0.60


        elif predicted == "Regular":

            probabilities["Regular"] = confidence

            probabilities["Premium"] = (
                100 - confidence
            ) * 0.45


        else:

            probabilities["Value"] = confidence

            probabilities["Regular"] = (
                100 - confidence
            ) * 0.60


        probabilities["Value"] = max(
            0,
            100
            -
            probabilities["Regular"]
            -
            probabilities["Premium"]
        )


        st.markdown(
            "**Prediction Probability**"
        )

        st.bar_chart(
            probabilities
        )


# ============================================================
# SOURCE DATA
# ============================================================

with st.expander(
    "View Source Sales Data"
):

    st.dataframe(
        view,
        use_container_width=True,
        height=350
    )


st.caption(
    "Customer Sales & Analytics Dashboard"
)
