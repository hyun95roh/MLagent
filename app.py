import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import statsmodels as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def llama_agent(task_description, stats_summary):
    prompt = (
        f"Task: {task_description}\n"
        f"Data Stats: {stats_summary}\n"
        f"Suggest an ML model suitable for the task and explain why. Also, mention remedies if any assumptions are violated."
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data['feature'] = features
    vif_data['VIF'] = [
        sm.OLS(df[feature], sm.add_constant(df[features].drop(columns=[feature]))).fit().rsquared
        for feature in features
    ]
    vif_data['VIF'] = 1 / (1 - vif_data['VIF'])
    return vif_data

def check_homoscedasticity(X, y):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    _, pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
    return pval

def check_autocorrelation(X, y):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    dw_stat = durbin_watson(model.resid)
    return dw_stat

st.set_page_config(page_title="ML Model Recommender Agent")
st.title("ML Model Recommender Agent")

st.markdown("""
Upload your dataset and describe your ML task in natural language. This agent will:
- Analyze the data format
- Perform statistical diagnostics (multicollinearity, heteroscedasticity, autocorrelation)
- Recommend suitable ML models with rationale
""")

# 1. User Task Description
nl_task = st.text_input("🧠 Describe your ML task")

# 2. File Upload
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # 3. Feature Selection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_features = st.multiselect("🔧 Select numeric features for analysis", options=numeric_cols, default=numeric_cols)

    if len(selected_features) >= 2:
        st.subheader("📈 Correlation Matrix")
        corr = df[selected_features].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # 4. VIF Calculation
        st.subheader("📉 Variance Inflation Factor (VIF)")
        try:
            vif = compute_vif(df, selected_features)
            st.dataframe(vif)
        except:
            st.warning("VIF computation failed. Ensure no NaNs and at least 2 features selected.")

        # 5. Additional Assumption Checks
        st.subheader("🧪 Other Assumption Checks")
        target = st.selectbox("Select target column for residual diagnostics", options=numeric_cols)

        if target and target in df.columns:
            X = df[selected_features].drop(columns=[target], errors='ignore')
            y = df[target]

            homo_pval = check_homoscedasticity(X, y)
            auto_dw = check_autocorrelation(X, y)

            st.markdown(f"**Breusch–Pagan test (Homoscedasticity p-value):** {homo_pval:.4f} — {'✔️ OK' if homo_pval > 0.05 else '⚠️ Heteroscedastic'}")
            st.markdown(f"**Durbin-Watson statistic (Autocorrelation):** {auto_dw:.4f} — {'✔️ OK' if 1.5 < auto_dw < 2.5 else '⚠️ Possible autocorrelation'}")

        # 6. Agent Suggestion
        if st.button("🦙 Suggest Model"):
            stats_summary = f"Columns: {selected_features}, VIF: {vif['VIF'].max():.2f}, Homoscedastic p={homo_pval:.4f}, Durbin-Watson={auto_dw:.2f}"
            suggestion = llama_agent(nl_task, stats_summary)
            st.success(suggestion)
    else:
        st.info("Please select at least 2 numeric features to analyze multicollinearity.")
else:
    st.info("Awaiting file upload...")
