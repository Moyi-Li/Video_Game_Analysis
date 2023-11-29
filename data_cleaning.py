"""
CSE 163 Group Project
Author: Moyi Li; JingjingDong; Zhikai Li
This program contains functions for reading and reprocessing the
datasets for further analysis in our final project.
The functions get_df and get_test can get the original datasets,
while the function get_final_result passed the final dataset
that have merged two dataframes df and test and after tiny adjustments.
"""
import pandas as pd


def get_df() -> pd.DataFrame:
    """
    This function reads the csv file of vgsales and
    returns the file in the dataframe format.
    """
    return pd.read_csv('vgsales.csv')


def get_test() -> pd.DataFrame:
    """
    This function reads the csv file of small_ds_test and
    returns the file in the dataframe format.
    """
    return pd.read_csv('small_ds_test.csv')


def get_final_result() -> pd.DataFrame:
    """
    This function reads the csv file of metacritic_games, drops the
    unnecessary columns, merges the vgsales and metacritic_games and
    returns the file in the dataframe format.
    """
    df = get_df()
    ratings = pd.read_csv('metacritic_games.csv')
    ratings_drop = ratings.drop(['release_date',  'summary'], axis=1)
    final_result = df.merge(ratings_drop, left_on=['Name', 'Platform'],
                            right_on=['name', 'platform'], how='inner')
    final_result.to_csv('final_result.csv', index=False)
    return final_result


def main():
    get_df()
    get_test()
    get_final_result()


if __name__ == '__main__':
    main()