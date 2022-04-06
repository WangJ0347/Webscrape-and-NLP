import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def data() -> pd.DataFrame:
    """
    :return: dataframe of 150 companies
    """
    data1 = pd.read_csv('fake_companies.csv')
    data2 = pd.read_csv('fake_companies_1.csv')
    data3 = pd.read_csv('webscrape_companies.csv')

    data3.columns = ['Unnamed: 0', 'Name', 'Purpose']

    combine_data = pd.concat([data1,data2,data3], axis=0)

    return combine_data


def best_worst_comps(comp: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze best and worst companies based on sentiment index
    :param comp:
    :return:2 DataFrames of 10 best and 10 worst companies
    """
    sia = SentimentIntensityAnalyzer()
    comp['Sentiment'] = comp['Purpose'].map(lambda x: sia.polarity_scores(x)['compound'])
    sorted_df = comp.sort_values(by='Sentiment')
    worst = sorted_df[:10]
    best = sorted_df[-1:-11:-1]

    return f"""best: 
    {best}
worst: 
{worst}"""


if __name__ == "__main__":
    a = data()
    b = best_worst_comps(a)
    print(b)