import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

column_names = ['Number', 'Topic', 'Sentiment', 'Comment']
df = pd.read_csv('twitter_training.csv', names=column_names)
df = df.dropna()

print(len(df['Topic'].unique()))

df2 = {'Topics': [df['Topic'].unique()], 'Negative': [x for x in df['Sentiment'] if x.index ]}



print(df.head().keys())