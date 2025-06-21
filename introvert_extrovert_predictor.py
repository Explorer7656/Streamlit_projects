import streamlit as st
import pandas as pd
import pickle as plk
import sklearn
from sklearn.pipeline import Pipeline

df=pd.read_csv(r'personality_dataset.csv')
# st.title('Are you an introvert or extrovert?')
# st.header('This is not a questionare, it is powered by a machine-learning regression model.')
with open(r'introvert_extrovert.pkl', 'rb') as file:
    model=plk.load(file)
print(df.columns.tolist())
print(df['Time_spent_Alone'].value_counts())
print(df['Stage_fear'].value_counts())
print(df['Social_event_attendance'].value_counts())
print(df['Going_outside'].value_counts())
print(df['Drained_after_socializing'].value_counts())
print(df['Friends_circle_size'].value_counts())
print(df['Post_frequency'].value_counts())
print(df['Personality'].value_counts())
print(model.classes_)


st.title('Find out are you an INTROVERT or EXTROVERT!')

time_alone=st.slider('How many hours alone per day:', min_value=0, max_value=11, step=1)
s=st.radio('Do you have stage fear?', ('Yes', 'No'))
if s=='Yes':
    stage_fear=1
else:
    stage_fear=0
social_events=st.slider('Rate how often do you attend social events:', min_value=0, max_value=10, step=1)
outside=st.slider('How many hour a day do you spend outside?', min_value=0, max_value=7, step=1)
d=st.radio('Are you drained after socializing?', ('Yes', 'No'))
if d=='Yes':
    drained=1
else:
    drained=0
friend_size=st.slider('How many friends do you have?', max_value=15, min_value=0)
post_frequency=st.slider('How often do you post on social media from 0 to 10?', min_value=0, max_value=10)
df2=pd.DataFrame({
    'Time_spent_Alone':[time_alone],
    'Stage_fear':[stage_fear],
    'Social_event_attendance':[social_events],
    'Going_outside':[outside],
    'Drained_after_socializing':[drained],
    'Friends_circle_size':[friend_size],
    'Post_frequency':[post_frequency]


})
if st.button('Get result'):
    prediction=model.predict(df2)[0]
    if prediction==1:
        st.success('You are an introvert!')
    else:
        st.success('You are an extrovert!')




