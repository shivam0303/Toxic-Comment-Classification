import twint
import pandas as pd
import os 
import numpy as np

class scrape_twitter:
    def __init__(self):
        pass

    def scrape_tweets_by_user(self,username,count):
        c = twint.Config()
        c.Username = username
        c.Limit = count
        c.Store_csv = True
        c.Output = "tweets.csv"
        twint.run.Search(c)
        df = pd.read_csv("tweets.csv")
        df.rename(columns={'tweet':'comment_text'},inplace=True)
        os.remove("tweets.csv")
        return df
