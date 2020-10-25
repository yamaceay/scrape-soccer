# import libs
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dateutil import parser
from datetime import datetime, timedelta
from warnings import filterwarnings as warner

warner("ignore")
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt

plt.style.use("ggplot")
import re
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import os

dirname = os.path.dirname(os.path.abspath(__file__))


def find_driver():
    if os.environ.get("CHROMEDRIVER_PATH") and os.environ.get("GOOGLE_CHROME_BIN"):
        env = True
    else:
        env = False

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option("prefs", {'profile.managed_default_content_settings.javascript': 2})

    if env:
        chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
        executable_path = os.environ.get("CHROMEDRIVER_PATH")
    else:
        executable_path = os.path.join(dirname, "chromedriver.exe")

    return webdriver.Chrome(executable_path=executable_path, chrome_options=chrome_options)


class tqdm:
    def __init__(self, iterable, title=None):
        if title:
            st.write(title)
        self.prog_bar = st.progress(0)
        self.iterable = iterable
        self.length = len(iterable)
        self.i = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)


# file manipulations
def find_csv():
    for file in os.listdir(dirname):
        if file.endswith(".csv"):
            return file[:-4]


def remove_csv(date):
    os.remove(dirname + "/" + date + ".csv")


date_of_df = find_csv()
df = pd.read_csv(date_of_df + ".csv").drop("Unnamed: 0", 1)

# app init
st.title("Team Mood Soccer")
st.header("This website is where you can track your favorite team's recent performance.")


# functions defined
def form_to_df():
    form_dfs = []
    league_names = []
    league_abbrs = []
    driver.get("https://www.soccerstats.com")
    for i in tqdm(range(1, 31), title="Waiting for new data..."):
        path = "//*[@id='headerlocal']/div[2]/table/tbody/tr/td[" + str(i) + "]/span/a"
        elem = driver.find_element_by_xpath(path)
        league_abbr = elem.get_attribute("innerText")
        league_name = re.findall(' alt="(.*?)" ', elem.get_attribute("outerHTML"))[0]
        elem.click()
        table = driver.find_element_by_xpath("/html/body/div/div/div[1]/div[2]/div[5]/div[3]/table[4]")
        form_df = pd.concat(pd.read_html(table.get_attribute("innerHTML")), axis=0)
        league_names.append(league_name)
        league_abbrs.append(league_abbr)
        form_dfs.append(form_df)
        driver.get("https://www.soccerstats.com")
    league_offsets = [len(x) for x in form_dfs]
    names = []
    abbrs = []
    forms = pd.concat(form_dfs, axis=0)
    for i in range(len(league_offsets)):
        for x in range(league_offsets[i]):
            names.append(league_names[i])
            abbrs.append(league_abbrs[i])
    forms["league_name"] = names
    forms["league_abbr"] = abbrs
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    forms["home"] = forms.loc[:, 1].apply(lambda x: x.split(" - ")[0])
    forms["away"] = forms.loc[:, 1].apply(lambda x: x.split(" - ")[1])
    forms = forms.drop(1, axis=1)
    forms = forms[forms[2] != "-"]
    forms["home_goal"] = forms[2].apply(lambda x: x.split("-")[0])
    forms["away_goal"] = forms[2].apply(lambda x: x.split("-")[1])
    forms = forms.drop(2, axis=1)
    forms["month"] = forms[0].map(lambda x: np.argmax([x.__contains__(y) for y in months]) + 1)
    forms["day"] = re.findall("\d+", " ".join(forms[0]))
    forms["str_day"] = ["0" + str(day) if len(str(day)) == 1 else str(day) for day in forms["day"]]
    forms["str_month"] = ["0" + str(month) if len(str(month)) == 1 else str(month) for month in forms["month"]]
    forms["datetime"] = forms["str_month"] + "/" + forms["str_day"]
    forms = forms.drop([0], 1)
    forms = forms[["datetime", "home", "away", "home_goal", "away_goal", "league_name", "league_abbr"]]
    forms_index = list(forms.index)
    count_match = "".join(
        [str(int(x.strip(" ")[-1]) + 1) for x in " ".join([str(x) for x in forms_index]).split("0 ")[1:]])
    matches_list = []
    offset = 0
    for count in list(count_match):
        n_match = int(count)
        lower = offset
        upper = lower + n_match
        offset += n_match
        matches = forms.iloc[lower:upper]
        matches_list.append(matches)
        team_names = []
    for teams in matches_list:
        team_names.append(pd.Series([*list(teams["home"].values), *list(teams["away"].values)]).value_counts().index[0])
    for i in range(len(team_names)):
        matches_list[i]["team"] = [team_names[i] for x in range(len(matches_list[i]))]
    df = pd.concat(matches_list, axis=0)
    df = df.reset_index(drop=True)
    df["opponent"] = [df["away"][i] if df["team"][i] == df["home"][i] else df["home"][i] for i in range(len(df))]
    df["avg_goal"] = df["home_goal"].astype(int) - df["away_goal"].astype(int)
    df = df.drop(["home", "away"], 1)
    return df


def shift_date(data):
    range_of_dates = np.arange(parser.parse("01/01/2019"), parser.parse("01/01/2020"), timedelta(days=1))
    range_of_dates = [*range_of_dates, *range_of_dates]
    range_of_dates = [parser.parse(str(x)).strftime("%m/%d") for x in np.datetime_as_string(range_of_dates)]

    locs = []
    length = len(data)
    data = sorted(data)
    for i in range(len(range_of_dates)):
        for x in data:
            if range_of_dates[i] == x:
                locs.append(i)
    diffs = [locs[i + 1] - locs[i] for i in range(length)]

    if np.argmax(diffs) == length - 1:
        locs = locs[:length]

    while np.argmax(diffs) < length - 1:
        locs = [locs[-1], *locs[:-1]]
        diffs = [abs(locs[i + 1] - locs[i]) for i in range(length)]

    start, end = locs[0], locs[-1]
    return range_of_dates[start:end + 1]


def plot_form(x, data=df):
    boca_juniors = data[["datetime", "team", "opponent", "league_abbr", "avg_goal"]][data["team"] == x].drop("team",
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
    for i in range(max(val_of_dates) - min(val_of_dates) + 2):
        length = len(dates)
        n = i + min(val_of_dates) - 1
        if n > 0:
            plt.plot(np.full(length, n), color="green", alpha=1)
        if n < 0:
            plt.plot(np.full(length, n), color="red", alpha=1)
        else:
            plt.plot(np.zeros(length), color="blue", alpha=0.2)
    return fig


if date_of_df != datetime.now().strftime("%m-%d"):
    driver = find_driver()
    df = form_to_df()
    df.to_csv(datetime.now().strftime("%m-%d") + ".csv")
    remove_csv(date_of_df)
    driver.quit()
else:
    df = pd.read_csv(date_of_df + ".csv").drop("Unnamed: 0", 1)

# perform data
league = st.sidebar.selectbox("Choose a league:", sorted(list(df["league_name"].value_counts().index)))
team = st.sidebar.selectbox("Choose a team:",
                            sorted(list(df[df["league_name"] == league]["team"].value_counts().index)))
st.header("Current mood of: " + team)
st.pyplot(plot_form(team))
