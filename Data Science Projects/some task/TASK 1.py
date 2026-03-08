import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('gender_submission.csv')
df = df.dropna()


print(df.head())

survival_counts = df['Survived'].value_counts()
print(survival_counts)

plt.bar(survival_counts.index, survival_counts.values)
plt.xticks([0, 1], ['Did not survive', 'Survived'])
plt.ylabel('Number of People')
plt.title('Survival Counts')
plt.tight_layout()
plt.show()