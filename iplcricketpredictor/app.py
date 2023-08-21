import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

teams=['Sunrisers Hyderabad','Mumbai Indians',
        'Royal Challengers Bangalore',
        'Kolkata Knight Riders',
        'Kings XI Punjab',
        'Chennai Super Kings',
        'Rajasthan Royals',
        'Delhi Capitals']

cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah','Mohali', 'Bengaluru']

pipe=pickle.load(open('pipe.pkl','rb'))
st.title(':orange[ IPL Win Predictor ]')
# st.title('A title with _italics_ :blue[colors] and emojis :sunglasses:')


col1,col2,col3=st.columns(3)
with col1:
    batting_team=st.selectbox('_:green[Select The Batting Team:]_',sorted(teams))
with col3:
    bowling_team=st.selectbox('_:green[Select The Bowling Team:]_',sorted(teams))
with col2:
    image=st.image('ipl image.png',width=200)

selected_cities=st.selectbox(':red[_Select host city:_]',sorted(cities))

target=(st.number_input(':red[_Target_]'))

col6,col4,col5=st.columns(3)
with col6:
    score=st.number_input(':blue[_Current Score_]')
with col4:
    overs=st.number_input(':blue[_Over Completed_]')
with col5:
    wickets=st.number_input(':blue[_Wickets out_]',)

if st.button('Predict Probability'):
    runs_left=(target-score)
    balls_left=(120-(overs*6))
    wicket=(10-wickets)
    try:
        crr=(score/overs)
        rrr=(runs_left*6)/overs
        input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_cities],
                                'total_runs_x':[target],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wicket],'crr':[crr],'rrr':[rrr]})
        # st.table(input_df)
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.header(batting_team + "- " + str(round(win*100)) + "%")
        st.header(bowling_team + "- " + str(round(loss*100)) + "%")
    except:
        st.header('fill the proper values',)

