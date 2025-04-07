import streamlit as st
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
st.dataframe(df)
data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.line_chart(data)
