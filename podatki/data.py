import numpy as np
import pandas as pd
import seaborn as sns 

def read_txt(ime):
    df = pd.read_table(ime, sep=',')
    df[' valid'] = pd.to_datetime(df[' valid'], format='%Y/%m/%d')
    df.sort_values(by=' valid', inplace = True) #urejanje po datumu
    df = df.rename(columns={'difuzno sev. [W/m2]': 'DF', ' valid': 'date'}) # preimenovanje stolpcev
    df['week_number'] = df['date'].dt.isocalendar().week # teden v letu
    del df["station id"] # izbris nepotrebnih stolpcev
    return df

vse_samodejne_2022 = read_txt('samodejne_postaje_2022.txt')
vse_samodejne_2021 = read_txt('samodejne_postaje_2021.txt')

# 2022
# sunek vetra
df_2022 = vse_samodejne_2022.loc[vse_samodejne_2022["sunki vetra [m/s]"] == 999.0, "sunki vetra [m/s]"] = np.nan
df_2022 = vse_samodejne_2022.astype({'sunki vetra [m/s]':'float'})

# negativno globalno sevanje
df_2022.loc[vse_samodejne_2022['globalno sev. [W/m2]'] < 0, 'globalno sev. [W/m2]']= np.nan

# postaje, ki imajo podatek za difuzno sevanje
df_2022 = df_2022[~df_2022['DF'].isna()]

# 2021
# sunek vetra
df_2021 = vse_samodejne_2021.loc[vse_samodejne_2021["sunki vetra [m/s]"] == 999.0, "sunki vetra [m/s]"] = np.nan
df_2021 = vse_samodejne_2021.astype({'sunki vetra [m/s]':'float'})

# negativno globalno sevanje
df_2021.loc[vse_samodejne_2021['globalno sev. [W/m2]'] < 0, 'globalno sev. [W/m2]']= np.nan

# postaje, ki imajo podatek za difuzno sevanje
df_2021 = df_2021[~df_2021['DF'].isna()]

data = pd.concat([df_2021, df_2022], ignore_index=True)
data.to_csv('data_df.csv', index=False)