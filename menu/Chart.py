import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_chart_page():
    # call df and filename
    if "df_replaced" in st.session_state:
        df = st.session_state.df_replaced
    elif "df" in st.session_state:
        df = st.session_state["df"]

# Def
    def Select_heatmap():
        target_folder = "Picture"

        # Create the target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        # Save the heatmap image to the target folder
        heatmap_file_path = os.path.join(target_folder, "heatmap.png")
        plt.figure(figsize=(20, 10))
        heatmap_chart = sns.heatmap(df.corr(), annot=True)
        heatmap_chart.get_figure().savefig(heatmap_file_path)

        # Display the image from the target location
        st.image(heatmap_file_path)
        st.markdown('---')

    def Select_pairplot(): 
        # Specify the target folder
        target_folder = "Picture"

        # Create the target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        # Save the pairplot image to the target folder
        pairplot_file_path = os.path.join(target_folder, "pairplot.png")
        pairplot_chart = sns.pairplot(data=df, hue='Potability')
        pairplot_chart.savefig(pairplot_file_path)

        # Display the image from the target location
        st.pyplot(pairplot_chart)
        st.markdown('---')

    def Select_boxplot():
        num_columns = 2  
        num_rows = (len(df.columns) + num_columns - 1) // num_columns

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 30))

        if num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_columns == 1:
            axes = axes.reshape(-1, 1)

        for i, column in enumerate(df.columns):
            row, col = divmod(i, num_columns)
            ax = axes[row, col]

            ax.boxplot(df[column].dropna())
            ax.set_title(f'Boxplot for {column}')
            ax.set_xlabel('__________________________________________________________________________________________________')

        st.pyplot(fig) 
        st.markdown('---')

    # Name Title 
    st.markdown('# Water Queality')

    # Checkbox Heatmap Pairplot Boxplot
    col1,col2,col3 = st.columns(3)

    with col1:
        heat_map = st.checkbox("HeatMap",True)

    with col2:
        See_pairplot = st.checkbox("Pairplot")

    with col3:
        See_boxplot = st.checkbox("Boxplot")
    st.markdown('---')

    if heat_map:
        Select_heatmap()
    if See_pairplot:
        Select_pairplot()
    if See_boxplot:
        Select_boxplot()

