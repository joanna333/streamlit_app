import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(layout="wide")

st.title("Worldwide Analysis of Quality of Life and Economic Factors")
st.write("""
This app enables you to explore the relationships between poverty, 
    life expectancy, and GDP across various countries and years. 
    Use the panels to select options and interact with the data.""")
tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

df = pd.read_csv("global_development_data.csv")
with tab1:
    st.header("Global Overview - Key Statistics")
    year = st.slider("Select Year for Visualisation", 1990, 2016)
    filtered_data = df[df["year"] == year]
    col1, col2, col3, col4 = st.columns(4)
    col1.write("Mean of life expectancy")
    mean_life_expectancy = filtered_data["Life Expectancy (IHME)"].mean()
    col1.subheader(f'{mean_life_expectancy:.2f}')
    col2.write("Global median of GDP per capita")
    medianGPT = filtered_data["GDP per capita"].median()
    col2.subheader(f'${medianGPT:,.0f}')
    col3.write("Global Poverty Average")
    headcount_ratio_upper_mid_income_povline_mean = filtered_data["headcount_ratio_upper_mid_income_povline"].mean()
    col3.subheader(f'{headcount_ratio_upper_mid_income_povline_mean:.0f}%')
    col4.write("Number of countries")
    num_countries = len(filtered_data["country"].unique())
    col4.subheader(f"{num_countries:.0f}")
    fig = px.scatter(filtered_data,
                     x="GDP per capita",
                     y="Life Expectancy (IHME)",
                     hover_name='country',
                     log_x=True,
                     size='Population',
                     color='country',
                     title=f'Life Expectancy vs GDP per capita ({year})',
                     labels={
                         'GDP per capita': 'GDP per Capita (USD)',
                         'Life Expectancy (IHME)': 'Life Expectancy (Years)'
                     })
    st.plotly_chart(fig)


    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    st.subheader("Predict Life Expectancy")
    st.write("This model uses timestamp, GDP per capita, and poverty rates to predict Life Expectancy.")
    st.write("Model Performance")
    mse = 6.214431392352935
    st.write(f"Mean squared error (MSE): {mse:.2f}")
    gdp = st.number_input("Enter the GDP per capita (dollars):", value=15000)
    pov = st.number_input("Enter the Poverty Rate (%):", value=30)
    year1 = st.number_input("Year of prediction:", value=2000)


    if st.button("Predict"):
        predictions = loaded_model.predict([[gdp, pov, year1]])
        st.write(f"Predicted Life Expectancy: {predictions[0]:.2f}")

    feature_importance = loaded_model.feature_importances_
    features = ['GDP per capita', 'Poverty Headcount Ratio', 'Year']

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(importance_df, x='Importance',  y='Feature', title="Feature Importance", orientation='h')
    st.plotly_chart(fig)



with tab3:
    st.title("Data Source")
    values = st.slider("Select Year Range", 1990, 2016, (1990, 2016))

    countries = st.multiselect(
        "Choose countries", list(df["country"].unique()), ["Germany"]
    )
    filtered_df = df[df["country"].isin(countries)]
    filtered_df = filtered_df[df["year"] >= values[0]]
    filtered_df = filtered_df[df["year"] <= values[1]]
    st.dataframe(filtered_df)

    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name='filtered_data.csv',
    )





