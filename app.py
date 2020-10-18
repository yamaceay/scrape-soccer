import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dateutil import parser
from datetime import datetime, timedelta
from warnings import filterwarnings as ignorer
ignorer("ignore")

df = pd.read_csv("latest.csv").set_index("Unnamed: 0", 1)

def shift_date(data):

    range_of_dates = np.arange(parser.parse("01/01/2019"), parser.parse("01/01/2020"), timedelta(days=1))
    range_of_dates = [*range_of_dates, *range_of_dates]
    range_of_dates = [parser.parse(str(x)).strftime("%m/%d") for x in np.datetime_as_string(range_of_dates)]

    locs = []
    length = len(data)
    data = sorted(data)
    for i in range(len(range_of_dates)):
        for x in data:
            if (range_of_dates[i] == x):
                locs.append(i)
    diffs = [locs[i + 1] - locs[i] for i in range(length)]

    if np.argmax(diffs) == length - 1:
        locs = locs[:length]

    while np.argmax(diffs) < length - 1:
        locs = [locs[-1], *locs[:-1]]
        diffs = [abs(locs[i + 1] - locs[i]) for i in range(length)]

    start, end = locs[0], locs[-1]
    return range_of_dates[start:end + 1]


def plot_form(x, df=df):
    boca_juniors = df[["datetime", "team", "opponent", "league_abbr", "avg_goal"]][df["team"] == x].drop("team",
                                                                                                         1).reset_index(
        drop=True)
    dates = shift_date(list(boca_juniors["datetime"]))
    val_of_dates = []
    opponents = []
    for i in range(len(dates)):
        if dates[i] in boca_juniors["datetime"].values:
            filtered_ = boca_juniors[boca_juniors["datetime"] == dates[i]]
            val = filtered_["avg_goal"].iloc[0]
            opponents.append(filtered_["opponent"].iloc[0])
        else:
            val = 0
            opponents.append("None")
        val_of_dates.append(val)
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_xticklabels(labels=dates, rotation=90)
    plt.scatter(dates, val_of_dates, marker="v", color="purple")
    for i in range(len(opponents)):
        if opponents[i] != "None":
            plt.annotate(opponents[i], [dates[i], val_of_dates[i]], rotation=90)
    plt.plot(boca_juniors["datetime"], boca_juniors["avg_goal"], color="black", alpha=0.3)
    for i in range(max(val_of_dates) - min(val_of_dates) + 1):
        length = len(dates)
        n = i + min(val_of_dates) - 1
        if (n > 0):
            plt.plot(np.full(length, n), color="green", alpha=1)
        if (n < 0):
            plt.plot(np.full(length, n), color="red", alpha=1)
        else:
            plt.plot(np.zeros(length), color="blue", alpha=0.2)
    return fig

st.title("Mood of Your Team?")

st.header("This website is where you can follow your favorite team's performance today.")

st.write(df[["datetime", "league_name", "team", "opponent", "avg_goal"]])

st.sidebar.text("TeamMood")

league = st.sidebar.selectbox("Choose a league:" , list(df["league_name"].value_counts().index))

team = st.sidebar.selectbox("Choose a team:", list(df[df["league_name"] == league]["team"].value_counts().index))

st.header("Current mood of: "+team)

st.pyplot(plot_form(team))