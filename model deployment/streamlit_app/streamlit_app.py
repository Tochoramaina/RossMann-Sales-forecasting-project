import streamlit as st 
import requests 
import numpy as np 
import pandas as pd 
import plotly.graph_objects as go 

st.set_page_config(page_title = 'Rossman Sales Forecast')
st.markdown('Handles **feature scaling** and **log transformations**')

store_id = st.sidebar.number_input('Store ID', min_value=0, max_value=1115, value=1)
time_steps = st.sidebar.slider("Historical Time steps (Days)", min_value=0, max_value=30, value=14)
num_features = 8

st.subheader('Historical input sequence matrix')
default_sequence = np.random.uniform(0, 1, (time_steps, num_features))
seq_df = pd.DataFrame(default_sequence, columns=[f'Feature_{i+1}' for i in range(num_features)])
edited_df = st.data_editor(seq_df)

if st.button("predict Sales"):
    payload = {
        "store_id": int(store_id),
        "sequence": edited_df.values.tolist()
    }
    with st.spinner('Calculating via fastapi Backend....'):
        try:
            res = requests.post("http://fastapi_app:8000/predict", json=payload)
            if res.status_code == 200:
                data = res.json()
                col1, col2 = st.columns(2)
                col1.metric("predicted logg_sales", f"{data['predicted_log_sales']:.4f}")
                col1.metric("Actual forecasted Sales", f"${data['predicted_sales']:.2f}")

                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = data['predicted_sales'],
                    title = {'text': f"Forecasted Sales for store {store_id}"},
                    number = {'prefix': "$"},
                    gauge = {'axis': {'range': [None, 20000]}}
                ))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f'error:{res.text}')
        except Exception as e:
            st.error(f'Failed to connect to Api: {e}')