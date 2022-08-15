import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4, 5],
    'second column': [10, 20, 30, 40, 50]
}))

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)


# sns.set_theme()
# -----------------------------------------------------------

# Helper functions
# -----------------------------------------------------------

df = pd.read_csv(
    "https://raw.githubusercontent.com/ThuwarakeshM/PracticalML-KMeans-Election/master/voters_demo_sample.csv"
)
sidebar = st.sidebar

n_clusters = sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
)


def run_kmeans(df, n_clusters=2):
    kmeans = KMeans(n_clusters, random_state=0).fit(df[["Age", "Income"]])

    fig, ax = plt.subplots(figsize=(16, 9))

    #Create scatterplot
    ax = sns.scatterplot(
        ax=ax,
        x=df.Age,
        y=df.Income,
        hue=kmeans.labels_,
        palette=sns.color_palette("colorblind", n_colors=n_clusters),
        legend=None,
    )

    return fig
# -----------------------------------------------------------

# MAIN APP
# -----------------------------------------------------------
# Show cluster scatter plot
st.write(run_kmeans(df, n_clusters=n_clusters))