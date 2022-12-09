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
popularity=pickle.load(open(r'C:/Users/DELL/Desktop/Book Recommendation/popularity.pkl','rb'))
user=pd.read_csv(r'C:/Users/DELL/Desktop/Book Recommendation/Users.csv')
book=pd.read_csv(r'C:/Users/DELL/Desktop/Book Recommendation/Books.csv')
rating=pd.read_csv(r'C:/Users/DELL/Desktop/Book Recommendation/Ratings.csv')


st.markdown(f'<h1 style="color: SkyBlue; font-family:Georgia, serif; font-weight: 800; font-size:40px;">{"Discover Books that you will love 📚📒📔"}</h1>', unsafe_allow_html=True)


st.markdown(f'<h3 style="color: lightblue;font-family:Verdana, sans-serif ; font-size:12px;">{"Enter a book you like and the site will analyze our huge database of real readers favorite books to provide book recommendations and suggestions for what to read next"}</h2>', unsafe_allow_html=True)

# For latest and tranding
book_list = dataset1.index.values
image_url = popularity['Image-URL-M'].tolist()
book_title = popularity['Book-Title'].tolist() 
book_author = popularity['Book-Author'].tolist()




#Building KNN model
x=dataset1.groupby('User-ID').count()['Book-Rating']>100

users_100=x[x].index

filtered_rating=dataset1[dataset1['User-ID'].isin(users_100)]
                         
y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=5
famous_books=y[y].index
final_rating=filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
      
final=final_rating.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating') 
pd.set_option('Display.max_columns',None)

final=final.fillna(0)


book_sparse=csr_matrix(final)

model=NearestNeighbors(algorithm="brute")

model.fit(book_sparse)

books_name=final.index


            
# fetching info
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
    book_id=np.where(final.index==book_name)[0][0]
    distance,suggestion=model.kneighbors(final.iloc[book_id,:].values.reshape(1,-1),n_neighbors=5)

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
m=100    #Set min number of votes
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

def bg_from_local(Desk):
    with open(Desk, "rb") as Desk:
        encoded_string = base64.b64encode(Desk.read())
    st.markdown(
    f"""
    <style>
    .stsidebar {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size:cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
bg_from_local('C:/Users/DELL/Desktop/Book Recommendation/flower-4151900.jpg')







#Functions for each of the pages

def page1(popularity):
    st.markdown(f'<h2 style="color:white ;font-family:Verdana, sans-serif; font-size:25px;">{"Latest & Trending Books:"}</h2>', unsafe_allow_html=True)
    
    
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
    st.markdown(f'<h2 style="color:white ;font-family:Verdana, sans-serif; font-size:25px;">{"Similar Books:"}</h2>', unsafe_allow_html=True)
    selected_books = st.selectbox("  ",books_name)
    if st.button('Show Recommendation'):
        recommended_book,poster_url = recommend_book(selected_books)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col2:
            st.text(recommended_book[0])
            st.image(poster_url[0])

        with col3:
            st.text(recommended_book[1])
            st.image(poster_url[1])
        
        with col4:
            st.text(recommended_book[2])
            st.image(poster_url[2])
        
        with col5:
            st.text(recommended_book[3])
            st.image(poster_url[3])
        
        with col1:
           st.text(recommended_book[4])
           st.image(poster_url[4])    
        
def Page3(df_author):
    st.markdown(f'<h2 style="color:white ;font-family:Verdana, sans-serif; font-size:25px;">{"Books of same author:"}</h2>', unsafe_allow_html=True)
    selected_author=st.selectbox('       ',unique_authors)
    suggested_books=author_based_recommender(selected_author)
    book_name,cover_Photo=display(suggested_books)
    Book_popular(book_name[:6],cover_Photo[:6])



#Sidebar navigation
    
option=st.sidebar.radio('Select what you want to display:',['Latest & Trending Books','Similar Books','Books of same author'])



#Navigation options
 
if option == 'Latest & Trending Books':
    page1(popularity) 
elif option == 'Similar Books':
    Page2(books_name)
elif option == 'Books of same author':
    Page3(df_author)

from PIL import Image
image = Image.open('C:/Users/DELL/Desktop/Book Recommendation/WhatsApp Image 2022-12-06 at 4.21.41 PM.jpeg')
st.sidebar.image(image)




