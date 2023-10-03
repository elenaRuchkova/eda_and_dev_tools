import pandas as pd

def main():
    df = pd.read_csv('./row/online_shoppers_intention.csv')

    df['Month'] = df['Month'].replace('aug','Aug')
    df['Informational_Duration'].fillna(df['Informational_Duration'].median(), inplace=True)
    df['ProductRelated_Duration'].fillna(df['ProductRelated_Duration'].median(), inplace=True)
    df['ExitRates'].fillna(df['ExitRates'].median(), inplace=True)
    df['Revenue'].replace({False: 0,
                           True: 1},inplace = True)

    df.to_csv('./row/online_shoppers_inention_eda.csv')



if __name__=='__main__':
    main()
    