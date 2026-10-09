import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="ChurnGuard Pro - Customer Churn Prediction", layout="wide")

st.title("🛡️ ChurnGuard Pro")
st.markdown("### Predict and prevent customer churn for your subscription or e-commerce business.")

st.sidebar.header("Upload Data")
st.sidebar.markdown("Upload your customer activity CSV file to scan for churn risks.")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.subheader("Uploaded Customer Dataset Preview")
    df_preview = pd.read_csv(uploaded_file)
    st.dataframe(df_preview.head(), use_container_width=True)
    
    if st.button("Run Churn Analysis", type="primary"):
        # Reset file pointer to beginning before sending to API
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        
        with st.spinner("Analyzing customer behavior patterns..."):
            try:
                # Send file to your FastAPI backend running locally
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                
                if response.status_code == 200:
                    result_data = response.json()
                    total_analyzed = result_data["total_rows_analyzed"]
                    predictions = result_data["predictions"]
                    
                    st.success(f"Successfully analyzed {total_analyzed} customer profiles!")
                    
                    # Convert results back to a DataFrame for display
                    res_df = pd.DataFrame(predictions)
                    
                    # Metrics overview
                    high_risk_count = (res_df['Predicted_Churn'] == 1).sum()
                    col1, col2 = st.columns(2)
                    col1.metric("Total Customers Scanned", total_analyzed)
                    col2.metric("High-Risk Churn Alerts", high_risk_count, delta_color="inverse")
                    
                    st.subheader("Detailed Risk Report")
                    st.dataframe(res_df, use_container_width=True)
                    
                    # Download button for results
                    csv_download = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Risk Report (CSV)",
                        data=csv_download,
                        file_name="churn_risk_report.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend API. Make sure your FastAPI server is running (`python -m uvicorn main:app --reload`).")
else:
    st.info("👈 Please upload a customer CSV file via the sidebar to begin.")