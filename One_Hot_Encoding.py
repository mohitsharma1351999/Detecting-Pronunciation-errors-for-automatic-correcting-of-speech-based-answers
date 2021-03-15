import pandas as pd
import numpy as np

audio_info = pd.read_csv('audio_info.csv')
df_lable = audio_info['lable']
df_lable = list(df_lable)
df_lable = np.array(df_lable)
df_lable

from sklearn.preprocessing import LabelEncoder

### integer mapping using LabelEncoder
encoded_value = LabelEncoder()

integer_encoded = encoded_value.fit_transform(df_lable)
integer_encoded.dump('label.pkl')

