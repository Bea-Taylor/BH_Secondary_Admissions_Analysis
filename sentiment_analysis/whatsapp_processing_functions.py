import numpy as np
import regex
import pandas as pd
import emoji

from collections import Counter

from transformers import BertTokenizerFast, pipeline

def date_time(s):
    "Returns true if the string is a date"
    pattern = '([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)'
    result = regex.search(pattern, s)
    if result:
        return True
    return False


def find_author(s):
    "Splits the string to find the author"
    s = s.split(":")
    if len(s)>=2:
        return True
    else:
        return False


def getDatapoint(line):
    "Extracts the date, time, author and message from a line - which is a single message"
    splitline = line.split('-')
    dateTime = splitline[0]
    date, time = dateTime.split(", ")
    date = regex.sub('\[', '', date) # added this line to help process errors in the date
    message = " ".join(splitline[1:])
    if find_author(message):
        splitmessage = message.split(":")
        author = splitmessage[0]
        message = " ".join(splitmessage[1:])
    else:
        author= None
    return date, time, author, message


def split_emoji(text):
    "Splits the text to find emojis"
    emoji_list = []
    data = regex.findall(r'\X',text)
    for word in data:
        if any(char in emoji.EMOJI_DATA for char in word):
            emoji_list.append(word)
    return emoji_list


def whatsapptxt_to_df(conversation):
    "Converts the whatsapp text file to a pandas dataframe"
    data = []
    with open(conversation, encoding="utf-8") as fp:
        fp.readline()
        messageBuffer = []
        date, time, author = None, None, None
        while True:
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            if date_time(line):
                if len(messageBuffer) > 0:
                    data.append([date, time, author, ' '.join(messageBuffer)])
                messageBuffer.clear()
                date, time, author, message = getDatapoint(line)
                messageBuffer.append(message)
            else:
                messageBuffer.append(line)

    df = pd.DataFrame(data, columns=['Date', 'Time', 'Author', 'Message'])
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], format='mixed').dt.time
    df['emoji'] = df["Message"].apply(split_emoji)

    return df


def extract_sentiment(df):
    "Extracts the sentiment from the messages and saves it in a new column"
    classifier = pipeline("text-classification", model="j-hartmann/sentiment-roberta-large-english-3-classes", top_k=None)
    sentiment = classifier(list(df['Message']))
    sentiment_df = pd.DataFrame(sentiment)
    df = pd.concat([df, sentiment_df], axis=1)

    return df 