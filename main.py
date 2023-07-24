import streamlit as st
import nltk
import glob
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import altair as alt

nltk.download('vader_lexicon')

st.title("Diary Tone")
st.subheader("Negativity")

directory = "diary"
txt_files = []
# Getting all sentiment scores for data
for filename in glob.iglob(f"{directory}/*"):
    with open(filename) as f:
        contents = f.read()
        txt_files = contents
        analyzer_a = SentimentIntensityAnalyzer()
        scores = analyzer_a.polarity_scores(txt_files)

        # -------------NEGATIVE VALUES-------------
        # Collecting negative score values
        d_neg = list(scores.values())[0]
        # Converting data from string to the datetime object
        filename = filename[6:]
        date = filename[:10]
        date_time_str = date
        date_time_obj = datetime.strptime(date_time_str,
                                          "%Y-%m-%d").date()
        # collecting data into the dictionary for a dataframe
        data = {
            "Date": [date_time_obj],
            "Negativity": [d_neg]
        }
        df = pd.DataFrame([data])

        # -------------POSITIVE VALUES-------------
        # Collecting positive score values
        d_pos = list(scores.values())[2]
        data_pos = {
            "Date": [date_time_obj],
            "Positivity": [d_pos]
        }
        df_pos = pd.DataFrame(data_pos)

    # Reading earlier created csv file (commented the csv append method bellow)
    data_new = pd.read_csv("data.csv", names=("Date", "Negativity"))
    line_chart = alt.Chart(data_new).mark_line().encode(
        x="Date",
        y="Negativity"
    )
    data_new_pos = pd.read_csv("data_pos.csv", names=("Date", "Positivity"))
    line_chart_pos = alt.Chart(data_new_pos).mark_line().encode(
        x="Date",
        y="Positivity"
    )

st.altair_chart(line_chart, use_container_width=True)
st.subheader("Positivity")
st.altair_chart(line_chart_pos, use_container_width=True)


# df.to_csv("data.csv", mode="a",
#           index=False, header=False)



