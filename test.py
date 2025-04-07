import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Business Analytics Dashboard", layout="wide")
st.sidebar.title("Upload & Filter Data")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

@st.cache_data

def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

if uploaded_file:
    try:
        df = load_data(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)


        # Identify column types
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'date' in df.columns.str.lower().tolist():
            df['Date'] = pd.to_datetime(df['Date'])

        st.sidebar.subheader("Filter Options")
        if categorical_cols:
            for col in categorical_cols:
                options = st.sidebar.multiselect(f"Filter by {col}", df[col].unique(), default=df[col].unique())
                df = df[df[col].isin(options)]

        if 'Date' in df.columns:
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
            df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

        # Data Exploration Section
        st.subheader("Basic Statistics")
        stats_col = st.multiselect("Select columns for statistics", numerical_cols, default=numerical_cols[:2])
        if stats_col:
            st.write(df[stats_col].describe())

        st.subheader("Search & Sort")
        if 'Product' in df.columns:
            search_text = st.text_input("Search Product")
            if search_text:
                df = df[df['Product'].str.contains(search_text, case=False)]
            if 'Sales' in df.columns:
                top_n = st.slider("Show Top N Best Selling Products", min_value=1, max_value=20, value=5)
                sorted_df = df.sort_values(by='Sales', ascending=False).head(top_n)
                st.dataframe(sorted_df)

        # Visualization Section
        st.sidebar.title("Create a Chart")
        chart_type = st.sidebar.selectbox("Chart Type", ["Bar", "Line", "Pie", "Scatter"])

        x_axis = st.sidebar.selectbox("X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Y-axis", numerical_cols)

        threshold = st.sidebar.slider("Minimum Y value", min_value=float(df[y_axis].min()), max_value=float(df[y_axis].max()), value=float(df[y_axis].min()))
        filtered_df = df[df[y_axis] >= threshold]

        st.subheader(f"{chart_type} Chart")
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "Bar":
            sns.barplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)
        elif chart_type == "Line":
            sns.lineplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax, marker='o')
        elif chart_type == "Pie":
            pie_data = filtered_df.groupby(x_axis)[y_axis].sum()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
            ax.axis('equal')
        elif chart_type == "Scatter":
            sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue=x_axis, ax=ax)

        st.pyplot(fig)

        # Export Section
        st.sidebar.title("Export Options")
        export_df = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Download Filtered Data", data=export_df, file_name="filtered_data.csv", mime="text/csv")

        # Optional advanced features - Machine Learning placeholder
        #with st.expander("Advanced Features: Predict Sales with Linear Regression"):
         #   if len(numerical_cols) >= 2:
               # ml_x = st.selectbox("Select Feature (X)", numerical_cols)
               # ml_y = st.selectbox("Select Target (Y)", numerical_cols, index=1)

               # X = df[[ml_x]]
               # y = df[ml_y]
               # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
               # model = LinearRegression()
               # model.fit(X_train, y_train)
                #predictions = model.predict(X_test)

                #st.write("### Model Performance")
                #st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")
                #st.write(f"R² Score: {r2_score(y_test, predictions):.2f}")

                #fig_ml, ax_ml = plt.subplots()
                #ax_ml.scatter(X_test, y_test, color='blue', label='Actual')
                #ax_ml.plot(X_test, predictions, color='red', linewidth=2, label='Predicted')
                #ax_ml.set_xlabel(ml_x)
                #ax_ml.set_ylabel(ml_y)
                #ax_ml.legend()
                #st.pyplot(fig_ml)

    except Exception as e:
     st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a dataset to begin.")
