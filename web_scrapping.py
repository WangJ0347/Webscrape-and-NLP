import pandas
import requests
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd


def get_req(url: str) -> str:
    """
    Perform get request
    :param url:
    :return: html string
    """
    recr = requests.get(url)
    status_code = recr.status_code
    if status_code != 200:
        raise ValueError("Status is not 200")
    else:
        return recr.text


def get_company(url: str) -> pandas.DataFrame:
    """
    This function return a Series of 50 companies with their purposes
    :param url: string of url
    :return: DataFrame
    """
    company = {}

    for i in range(50):
        info = get_req(url)
        company[re.findall("Name: (.*)<", info)[0]] = re.findall("Purpose: (.*)<", info)[0]

    df = pd.DataFrame()
    df['Name'] = company.keys()
    df['Purpose'] = company.values()

    return df


def best_worst_comps(comp: pandas.DataFrame) -> pandas.DataFrame:
    """
    Analyze best and worst companies based on sentiment index
    :param comp:
    :return: 2 DataFrames of 10 best and 10 worst companies
    """
    sia = SentimentIntensityAnalyzer()
    comp['Sentiment'] = comp['Purpose'].map(lambda x: sia.polarity_scores(x)['compound'])
    sorted_df = comp.sort_values(by='Sentiment')
    worst = sorted_df[:10]
    best = sorted_df[-1:-10:-1]

    return best, worst


if __name__ == "__main__":
    # comp = get_company('http://34.238.119.208:8000/random_company')
    # comp.to_csv('fake_companies.csv')
    a = pd.read_csv('fake_companies.csv')
    b = best_worst_comps(a)
    print(b)
