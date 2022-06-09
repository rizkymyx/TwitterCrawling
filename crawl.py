import tweepy
from textblob import TextBlob
import csv


api_key = "oMfjVFXDoVVjsU3C7cniPPw9u"
api_secret_key = "2vum5adycYLMGYQW6JRLZZb4u7J3eJBP4XeduAVwbncRH2odqJ"
access_token = "1446265685185761281-b2vDpmC1wXaqKspUs2UScPS5wy7gPP"
access_token_secret = "sRaAL9efaURXAgEOQJiDnbrVufCFCjCnq80XpOSDn9U1Z"

auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit = True)

csvFile = open('resultcrawl70.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

max_tweet = 4000
for tweet in tweepy.Cursor(api.search_tweets,
                           q = "online study OR pandemic -filter:retweets", result_type = "recent",
                           lang="en").items(max_tweet):
    
    analysis = TextBlob(tweet.text)
    


    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.text.encode('utf-8'), analysis.polarity])

csvFile.close()