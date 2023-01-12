import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit import session_state as session

def name_to_index(list_of_restaurants,dataframe_with_name):
    index_list =[]
    for string_test in list_of_restaurants:
        res_index=dataframe_with_name[dataframe_with_name['name'].str.contains(string_test)].index[0]
        index_list.append(res_index)
    return index_list

def scaling_by_rating(user_dict,unscaled_df,num_recommendations=10,state='all'):
    """
    This function aims to scale the column similarity based on the rating given to the restaurant 
    
    Input: The first argument is a dictionary made of the restaurants they've gone to and the ratings they've given
    them. The second argument is the unscaled dataframe made from user_creation. the next one is how many 
    recommendations to print out, and the last one is the state that they want to include
    
    type:
    user_dict: dictionary 
    unscaled_df: pandas dataframe
    num_recommendations: int
    state= string
    
    output: a pandas dataframe with rows size given by num_recommendations and potentially sorted by state 
    
    
    """
    #creating a copy of the unscaled df
    copy_df=unscaled_df.copy()
    
    #Sorting by Ontario or Nevada
    if state == 'ON' or state == 'NV':
        unscaled_df=unscaled_df[unscaled_df['state'].str.contains(state)]
    
    #Dropping unnecessary columns
    unscaled_df=unscaled_df.drop(columns=['name','state'])
    
    #making min -> max ranking change to max->min ranking 
    for i in range(0,len(unscaled_df.columns)):
        #getting the maximum 
        maximum=unscaled_df[unscaled_df.columns[i]].max()
        #swapping the max and min scale 
        unscaled_df[unscaled_df.columns[i]]=maximum-unscaled_df[unscaled_df.columns[i]]
        
    #creating a empty list 
    review_list=[]
    
    #Getting the review score
    for index in user_dict:
        review_list.append(user_dict[index])
        
        
    #asserting that the lengths are the same 
    if len(unscaled_df.columns)==len(review_list):
        #iterating over the review_list 
        for i in range(0,len(review_list)):
            #scaling the columns based on the review score 
            unscaled_df[unscaled_df.columns[i]]=review_list[i]*unscaled_df[unscaled_df.columns[i]]
    #if it is not equal then it will print the lengths of each
    else: print(len(review_list),len(unscaled_df.columns)) 
    
    #getting the top number of recommendations based on input
    top_x=unscaled_df.sum(axis=1).sort_values(ascending=False).head(num_recommendations)
    
    #merging the dataframe to get the names
    result_df=pd.merge(copy_df['name'],pd.DataFrame(top_x,columns=['Result']),left_index=True,right_index=True)
    
    #sorting by Result 
    result_df=result_df.sort_values(by='Result',ascending=False)
    
    #returning a dataframe 
    return result_df

def user_creation(user,algo,df_trained,df_name_stored):
    
    """
    Input: The first argument is a user, that is in the form of a dictionary. The second is the knn model that should 
    be fitted on a dataframe. The next is the dataframe that the KNN was trained on. The last is a dataframe 
    where the names are stored.
    Types:
    user:dictionary 
    algo:KNN model
    df_trained: pandas dataframe 
    df_name_stored: pandas dataframe 
    
    output: a pandas dataframe that will be the similarity between the input restaurants and the rest of the 
    restaurants. Values closer to 0 between restaurants indicate higher similarity 
    
    
    """
    
    #creating the user_df 
    user_df=df_name_stored[['name','state']]
    index_list=[]
    #for loop over the indexes in the user input
    for index in user:
        #getting the distances and indices from the knn 
        distances,indices=algo.kneighbors(df_trained[index:index+1],n_neighbors=df_trained.shape[0])
        
        #getting the names of the businesses 
        name_business=df_name_stored[df_name_stored.index==index]['name']
        
        #stripping the name of the businesses 
        name_business=name_business[index].strip('\"')
        
        #Creating column that are named by the name of the business + similarity with values based on similarity of
        #each business with the input business 
        similarity_df = pd.DataFrame(distances.flatten(),index=indices.flatten(),columns=[name_business+' Similarity'])
                
        #concatting the column into the user dataframe
        user_df=pd.concat((user_df,similarity_df),axis=1)
        index_list.append(index)
        
    user_df.drop(index_list,inplace=True)
        
        
        #returning the user df
    return user_df

st.set_page_config(
    page_title="Yelp Recommender System",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache(persist=True, show_spinner=False, suppress_st_warning=True)
def load_data():
    """
    load and cache data
    :return: tfidf data
    """
    PCA_dataframe = pd.read_csv("data/df_X_train_PCA.csv",index_col=0)
    return PCA_dataframe


df_X_train_PCA = load_data()

@st.cache(persist=True, show_spinner=False, suppress_st_warning=True)
def load_data2():
    """
    load and cache data
    :return: tfidf data
    """
    PCA_dataframe = pd.read_csv("data/df_final.csv", index_col=0)
    return PCA_dataframe


df_business_final = load_data2()
df_business_final=df_business_final.reset_index()
# load the model from disk
knn_model = pickle.load(open('knnpickle_file', 'rb'))

restaurant_list=list(df_business_final['name'])
restaurant_list.sort()
restaurant_list=[i.replace("\"","") for i in restaurant_list]
dataframe = None

st.title("""
Your Personalized Travel Food Guide
 """)
st.header('This is an Content-Based Recommender System for restaurants in Toronto and Las Vegas made from the Yelp 2018 Dataset. :sparkles:')


st.text("")
st.text("")
st.text("")
st.text("")

col2, col3 = st.columns(2)

with col2:
    st.subheader('This is where you input your favourite restaurants! :sushi: :pizza: :ramen: ')
    session.options = st.multiselect(label="Select Restaurants",options=restaurant_list)

    option = st.selectbox(
        'Where would you like recommendations?',
        ('Toronto', 'Las Vegas', 'Both'))

    if option == 'Toronto':
        state_choice='ON'
    elif option == 'Las Vegas':
        state_choice='NV'
    else:
        state_choice='all'

    num_recs = st.slider('How many recommendations would you like?', 5, 50, 10)


buffer1, col1, buffer2 = st.columns([0.5, 1, 1])
is_clicked = col1.button(label="Recommend")

if is_clicked:
    index_list=name_to_index(session.options ,df_business_final)
    ratings=np.arange(0,len(index_list))
    for i in ratings:
        ratings[i]=5

    user_dict={}

    for i in np.arange(0,len(index_list)):
        user_dict[index_list[i]]=[ratings[i]]
    array1=knn_model.kneighbors(df_X_train_PCA[0:1])
    print(array1)


    df_user=user_creation(user_dict,knn_model,df_X_train_PCA,df_business_final)
    with col3:
        st.subheader("These are your curated recommendations! I hope you enjoy :white_check_mark: ")
        dataframe=scaling_by_rating(user_dict,df_user,num_recommendations=num_recs,state=state_choice)
        st.table(dataframe)
