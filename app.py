import streamlit as st
from PIL import Image 
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Mi APP', layout="wide",
				   initial_sidebar_state="collapsed")

def main():
	st.title("Bienvenidos a Hate Speech Detection")
	st.sidebar.header("Navegación")
	df = pd.read_csv("labeled_data.csv")
	st.dataframe(df)
	img = Image.open("hatespeech.png")
	st.image(img, use_container_width=True)
	st.image("https://livewire.thewire.in/wp-content/uploads/2022/12/LW-cover-designs.gif", use_container_width=True)
	df_count = df.groupby('Gender').count().reset_index()
	fig = px.pie(df_count, values="hate_speech", names="ofensive_language", title="count")
	st.plotly_chart(fig)
	  

if __name__ == "__main__":
	main()
 
 