# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:31:32 2022

@author: DELL
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import requests 
import pickle
import plotly.express as px

# DATA FRAME
dataset1=pd.read_csv("C:/Users/DELL/Desktop/Book Recommendation/dataset1.csv")
model = pickle.load(open(r"C:/Users/DELL/Desktop/Book Recommendation/model.pkl", "rb"))
final= pickle.load(open(r"C:/Users/DELL/Desktop/Book Recommendation/final.pkl", "rb"))
book_name=pickle.load(open(r'C:/Users/DELL/Desktop/Book Recommendation/books_name.pkl','rb'))
popularity=pickle.load(open(r'C:/Users/DELL/Desktop/Book Recommendation/popularity.pkl','rb'))



st.markdown(f'<h1 style="color: SkyBlue;font-size:35px;">{"📚📒📔Discover Books that you will love!"}</h1>', unsafe_allow_html=True)

st.markdown(f'<h2 style="color: lightblue;font-size:20px;">{"Enter a book you like and the site will analyze our huge database of real readers favorite books to provide book recommendations and suggestions for what to read next"}</h2>', unsafe_allow_html=True)

# For latest and tranding
book_list = final.index.values
image_url = popularity['Image-URL-M'].tolist()
book_title = popularity['Book-Title'].tolist() 
book_author = popularity['Book-Author'].tolist()


# for collabrative based
def fetch_poster(suggestion):
    
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(final.index[book_id])

    for name in book_name[0]: 
        ids = np.where(dataset1['Book-Title'] == name)[0][0]
        ids_index.append(ids)
        
    for idx in ids_index:
        url = dataset1.iloc[idx]['Image-URL-M']
        poster_url.append(url)

    return poster_url
def recommend_book(book_name):
    books_list = []
    book_id = np.where(final.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(final.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
            books = final.index[suggestion[i]]
            for j in books:
                books_list.append(j)
    return books_list , poster_url

  
#Background image
import base64
def add_bg_from_local(Desk):
    with open(Desk, "rb") as Desk:
        encoded_string = base64.b64encode(Desk.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size:cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('C:/Users/DELL/Desktop/Book Recommendation/Desk.jpg')



#Functions for each of the pages

def page1(popularity):
    st.markdown(f'<h2 style="color: pink;font-size:30px;">{"Latest & Trending Books:"}</h2>', unsafe_allow_html=True)
    
    
    if st.button('Show Recommended book'):
        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
          st.image(image_url[0])
          st.text("Book Title:"+book_title[0])
          st.text("Author:"+book_author[0])
          
        with col2:
          st.image(image_url[1])
          st.text("Book Title:"+book_title[1])
          st.text("Author:"+book_author[1])
        
        with col3:
          st.image(image_url[2])
          st.text("Book Title:"+book_title[2])
          st.text("Author:"+book_author[2])
          
        with col4:
          st.image(image_url[3])
          st.text("Book Title:"+book_title[3])
          st.text("Author:"+book_author[3])
        
        with col5:
          st.image(image_url[4])
          st.text("Book Title:"+book_title[4])
          st.text("Author:"+book_author[4])
        

        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
          st.image(image_url[5])
          st.text("Book Title:"+book_title[5])
          st.text("Author:"+book_author[5])

        with col2:
          st.image(image_url[6])
          st.text("Book Title:"+book_title[6])
          st.text("Author:"+book_author[6])
     
        with col3:
          st.image(image_url[7])
          st.text("Book Title:"+book_title[7])
          st.text("Author:"+book_author[7])

        with col4:
          st.image(image_url[8])
          st.text("Book Title:"+book_title[8])
          st.text("Author:"+book_author[8])

        with col5:
          st.image(image_url[9])
          st.text("Book Title:"+book_title[9])
          st.text("Author:"+book_author[9])


        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
          st.image(image_url[10])
          st.text("Book Title:"+book_title[10])
          st.text("Author:"+book_author[10])

        with col2:
          st.image(image_url[11])
          st.text("Book Title:"+book_title[11])
          st.text("Author:"+book_author[11])

        with col3:
          st.image(image_url[12])
          st.text("Book Title:"+book_title[12])
          st.text("Author:"+book_author[12])

        with col4:
          st.image(image_url[13])
          st.text("Book Title:"+book_title[14])
          st.text("Author:"+book_author[13])

        with col5:
          st.image(image_url[14])
          st.text("Book Title:"+book_title[15])
          st.text("Author:"+book_author[14])


        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
          st.image(image_url[15])
          st.text("Book Title:"+book_title[15])
          st.text("Author:"+book_author[15])

        with col2:
          st.image(image_url[16])
          st.text("Book Title:"+book_title[16])
          st.text("Author:"+book_author[16])

        with col3:
          st.image(image_url[17])
          st.text("Book Title:"+book_title[17])
          st.text("Author:"+book_author[17])
   
        with col4:
          st.image(image_url[18])
          st.text("Book Title:"+book_title[18])
          st.text("Author:"+book_author[18])

        with col5:
          st.image(image_url[19])
          st.text("Book Title:"+book_title[19])
          st.text("Author:"+book_author[19])


        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
          st.image(image_url[20])
          st.text("Book Title:"+book_title[20])
          st.text("Author:"+book_author[20])

        with col2:
          st.image(image_url[21])
          st.text("Book Title:"+book_title[21])
          st.text("Author:"+book_author[21])

        with col3:
          st.image(image_url[22])
          st.text("Book Title:"+book_title[22])
          st.text("Author:"+book_author[22])

        with col4:
          st.image(image_url[23])
          st.text("Book Title:"+book_title[23])
          st.text("Author:"+book_author[23])

        with col5:
          st.image(image_url[24])
          st.text("Book Title:"+book_title[24])
          st.text("Author:"+book_author[24])
          
          
          

         
             
def Page2(book_name):
    
    selected_books = st.selectbox("  ",book_name)
    if st.button('Show Recommendation'):
        recommended_books,poster_url = recommend_book(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.text(recommended_books[0])
        st.image(poster_url[0])
        
    with col2:
        st.text(recommended_books[1])
        st.image(poster_url[1])

    with col3:
        st.text(recommended_books[2])
        st.image(poster_url[2])
        
    with col4:
        st.text(recommended_books[3])
        st.image(poster_url[3])
        
    with col5:
        st.text(recommended_books[4])
        st.image(poster_url[4])
 
    
#Sidebar navigation
st.sidebar.title('Navigation:')
 
option=st.sidebar.radio('Select what you want to display:',['Latest & Trending Books','User Based '])

#Navigation options 
if option == 'Latest & Trending Books':
  page1(popularity) 
elif option == 'User Based ':
  Page2(book_name)
