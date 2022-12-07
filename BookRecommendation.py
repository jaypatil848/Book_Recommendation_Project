# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:31:32 2022

@author: DELL
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
import warnings
warnings.filterwarnings('ignore')

# DATA FRAME
dataset1=pd.read_csv("C:/Users/DELL/Desktop/Book Recommendation/dataset1.csv")
model = pickle.load(open(r"C:/Users/DELL/Desktop/Book Recommendation/model.pkl", "rb"))
final= pickle.load(open(r"C:/Users/DELL/Desktop/Book Recommendation/final.pkl", "rb"))
books_name=pickle.load(open(r"C:/Users/DELL/Desktop/Book Recommendation/books_name.pkl","rb"))
popularity=pickle.load(open(r'C:/Users/DELL/Desktop/Book Recommendation/popularity.pkl','rb'))
user=pd.read_csv(r'C:/Users/DELL/Desktop/Book Recommendation/Users.csv')
book=pd.read_csv(r'C:/Users/DELL/Desktop/Book Recommendation/Books.csv')
rating=pd.read_csv(r'C:/Users/DELL/Desktop/Book Recommendation/Ratings.csv')



st.markdown(f'<h1 style="color: SkyBlue; font-family:Georgia, serif; font-weight: 800; font-size:40px;">{"Discover Books that you will love 📚📒📔"}</h1>', unsafe_allow_html=True)


st.markdown(f'<h3 style="color: lightblue;font-family:Verdana, sans-serif ; font-size:12px;">{"Enter a book you like and the site will analyze our huge database of real readers favorite books to provide book recommendations and suggestions for what to read next"}</h2>', unsafe_allow_html=True)

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
book_sparse=csr_matrix(final)
model=NearestNeighbors(algorithm="brute")
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


# Author Based Recommendation
def display(suggested_book):
    #Creating a books data set which does not have any duplicate books
    books=book.drop_duplicates(['Book-Title'])
    book_name=[]
    cover_Photo=[]
    for i in range(len(suggested_book)):
        #Gives a data frame with satisfy that condition
        filter1=books[books['Book-Title']==suggested_book[i]]
        book_name.append(filter1['Book-Title'].values[0])
        cover_Photo.append(filter1['Image-URL-L'].values[0])
    return book_name,cover_Photo


#Display 5 separate columns 
def Book_popular(book_name,cover_Photo):
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(cover_Photo[0],use_column_width=True,caption=book_name[0])
    with col2:
        st.image(cover_Photo[1],use_column_width=True,caption=book_name[1])
    with col3:
        st.image(cover_Photo[2],use_column_width=True,caption=book_name[2])
    with col4:
        st.image(cover_Photo[3],use_column_width=True,caption=book_name[3])
    with col5:
        st.image(cover_Photo[4],use_column_width=True,caption=book_name[4])


df_author=user.merge(rating,on='User-ID')
df_author=df_author.merge(book,on='ISBN')
df_author['Average_rating']=df_author.groupby('Book-Title')['Book-Rating'].transform('mean')
df_author['Number_of_rating']=df_author.groupby('Book-Title')['Book-Rating'].transform('count')
#Minimum number of vote
v=df_author['Number_of_rating']
R=df_author['Average_rating']
c=df_author['Average_rating'].mean()
m=200    #Set min number of votes
df_author['weighted_average']=((R*v)+(c*m))/(v+m)
unique_authors=df_author['Book-Author'].unique() 
df_author.drop(['Age','Location','Image-URL-S','Image-URL-M'],axis=1,inplace=True)
df_author.drop_duplicates('Book-Title',inplace=True)
df_author=df_author.sort_values('weighted_average',ascending=False)


def author_based_recommender(author_name):
    
    selected_df=df_author[df_author['Book-Author']==author_name]
    selected_df=selected_df.drop_duplicates('Book-Title')
    book_list=selected_df['Book-Title'].values
    book_list=list(book_list)
    return book_list





  
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
add_bg_from_local('C:/Users/DELL/Desktop/Book Recommendation/WhatsApp Image 2022-12-06 at 4.20.04 PM.jpeg')


#Functions for each of the pages

def page1(popularity):
    st.markdown(f'<h2 style="color: skin;font-family:Verdana, sans-serif; font-size:25px;">{"Latest & Trending Books:"}</h2>', unsafe_allow_html=True)
    
    
    if st.button('Show Books'):
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
          
          
          

         
             
def Page2(books_name):
    
    selected_books = st.selectbox("  ",books_name)
    if st.button('Show Recommendation'):
        recommended_books,poster_url = recommend_book(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
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
        
    with col1:
        st.text(recommended_books[5])
        st.image(poster_url[5])
#Sidebar navigation
st.sidebar.title('Navigation:')


#st.sidebar.image("")
 
option=st.sidebar.radio('Select what you want to display:',['Latest & Trending Books','Books Recommended for you ','Author'])

#Navigation options 
if option == 'Latest & Trending Books':
  page1(popularity) 
elif option == 'Books Recommended for you ':
  Page2(books_name)
elif option == 'Author':
  selected_author=st.selectbox('       ',unique_authors)
  suggested_books=author_based_recommender(selected_author)
  book_name,cover_Photo=display(suggested_books)
  Book_popular(book_name[:6],cover_Photo[:6])

from PIL import Image
image = Image.open('C:/Users/DELL/Desktop/Book Recommendation/WhatsApp Image 2022-12-06 at 4.21.41 PM.jpeg')
st.sidebar.image(image)

