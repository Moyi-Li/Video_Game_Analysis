"""
CSE 163 Group Project
Author: Moyi Li; JingjingDong; Zhikai Li
This program contains functions for analyzing the 5 different research
problems for video games using plotting, Machine learning, etc.
The dataset includes 16598 rows and 11 columns, including the name,
releasing platform, year, genre, publisher, sales in global and each region,
and rank of overall sales.
The second dataset Metacritic Games includes 19992 rows and 6 columns,
including variables name, platform, release time, a short summary
for each game, and meta_score and user_score for evaluating each game.
Both datasets are from Kaggle. In investigating the research questions,
we are going to merge these two dataset based on the names of the
video games to collect information for both
sales and scores for production excellence.
"""


import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

from data_cleaning import get_df, get_final_result

sns.set()


# Question 1: Pie chart
def pie_chart(final_result: pd.DataFrame) -> None:
    """
    This function takes in the merged dataset final_result, and
    create a pie chart to figure out the proportional distribution of
    genres in terms of their numbers of video games, and returns nothing.
    """
    genre_counts = final_result.groupby('Genre')['Name'].count()
    fig = go.Figure(data=[go.Pie(labels=genre_counts.index,
                    values=genre_counts)])
    fig.update_layout(title='Genre Distribution', xaxis_title='Genre',
                      yaxis_title='Name Count')
    fig.show()


# Question 2: Bar plot
def histogram_plot(final_result: pd.DataFrame) -> None:
    """
    This function takes in the merged dataset final_result, and
    identified the top 5 publisher that have published
    the highest number of video games.
    After that, created a proportional stacked bar plot to figure out
    how sales in each region vary for the top 5 publishers.
    """
    publisher_counts = final_result.groupby('Publisher')['Name'].count()
    top_publishers = publisher_counts.sort_values(
                    ascending=False).head(5).index
    print(top_publishers)

    # by checking the dataframe top_publishers,
    # we found out the top 5 publishers:
    five_publisher = ['Activision', 'Electronic Arts',
                      'Nintendo', 'Sega', 'Ubisoft']

    top_filtered = final_result[final_result['Publisher'].isin(five_publisher)]
    all_reigons = top_filtered.groupby(
                 'Publisher')[["NA_Sales", "EU_Sales",
                               "JP_Sales", "Other_Sales"]].sum().reset_index()

    all_reigons = pd.DataFrame(all_reigons)

    # this represent the proportional data that we would like to plot
    stacked_data = all_reigons.iloc[:, 1:].apply(lambda x:
                                                 x*100/sum(x), axis=1)

    ax = stacked_data.plot(kind='barh', stacked=True, color=[
        'mediumaquamarine', 'violet', 'gold', 'hotpink'])

    ax.set_yticklabels(five_publisher)

    plt.title("Region Sales for Top 5 Publishers")
    plt.xlabel('Sales Percentage of Each Region')
    plt.ylabel('Top 5 publishers')
    plt.savefig('histogram_plot.png', bbox_inches='tight')
    plt.show()


# problem 3: machine learning and model training
def fit_and_predict_sales(df: pd.DataFrame, hid_unit: int,
                          epoch_size: int, batch_size: int) -> float:
    """
    This function will return a test mean square error of the prediction of
    Global_Sales with the given data set and three integers that determine
    the number of neurons, the times to iterate over the training set, and
    the number of samples that is used each time.
    """
    df_filtered = df[['Platform', 'Year', 'Genre',
                      'Publisher', 'Global_Sales']]
    df_filtered = df_filtered.dropna()
    features = df_filtered.loc[:, df_filtered.columns != 'Global_Sales']
    features = pd.get_dummies(features)
    labels = df_filtered['Global_Sales']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)

    # For our challenge, we want to use a sequential model in keras.
    model = Sequential()
    # units is the neurons inside the dense layer.
    # relu activation converts the output to
    # a minimum of zero and unlimited upward value.
    # input_dim is to confirm the number of features in our input data.
    model.add(Dense(units=hid_unit,
                    input_dim=features_train.shape[1], activation='relu'))
    # This is the output layer for the model.
    model.add(Dense(1))
    # loss is the measure the differences between predicted and real output.
    # metrics are used to evaluate the performance of the model during training
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    # epochs is the number of times to iterate over the training dataset.
    # batch size is about number of samples that used by the model each time.
    # Larger batch size train faster but more risk of overfitting,
    # conversely is slower with more accuracy.
    model.fit(features_train, labels_train, epochs=epoch_size,
              batch_size=batch_size, verbose=2,
              validation_data=(features_test, labels_test))
    labels_pred = model.predict(features_test)
    output = mean_squared_error(labels_test, labels_pred)
    return output


def fit_and_predict_final_result(df: pd.DataFrame, hid_unit: int,
                                 epoch_size: int, batch_size: int) -> float:
    """
    This function will return a test mean square error of the prediction of
    Global_Sales with the given data set and three integers that determine
    the number of neurons, the times to iterate over the training set, and
    the number of samples that is used each time.
    """
    df_filtered = df[['Name', 'Platform', 'Year',
                      'Genre', 'Publisher', 'Global_Sales']]
    df_filtered = df_filtered.dropna()
    features = df_filtered.loc[:, df_filtered.columns != 'Global_Sales']
    features = pd.get_dummies(features)
    labels = df_filtered['Global_Sales']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)

    model = Sequential()
    model.add(Dense(units=hid_unit,
                    input_dim=features_train.shape[1], activation='relu'))
    # Dropout random part for preventing overfitting.
    model.add(Dropout(0.2))
    model.add(Dense(units=hid_unit/2, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model.fit(features_train, labels_train, epochs=epoch_size,
              batch_size=batch_size,
              verbose=2, validation_data=(features_test, labels_test))
    labels_pred = model.predict(features_test)
    output = mean_squared_error(labels_test, labels_pred)
    return output


# Question 4: regression chart
def regression_analysis(final_result: pd.DataFrame) -> None:
    """
    Takes in the merged dataset, calculates the coefficients
    of determination of the regression models of meta or user
    score and global sales, creates the regression line
    chart for the model. For clearer viewing, the data is
    filtered to a narrower range and the calcuation and plotting
    is created for both filtered and unfiltered data.
    """
    final_result = final_result.fillna(0)
    # using linear regression model to calculate the
    # coefficients of determination which measures how
    # the regression line performs.
    model = LinearRegression()
    meta_real_all = np.array(final_result['meta_score']).reshape(-1, 1)
    global_real_all = np.array(final_result['Global_Sales'])
    model.fit(meta_real_all, global_real_all)
    meta_all = (f"coefficients of determination for meta score "
                f"before filtering: "
                f"{model.score(meta_real_all, global_real_all)}")
    print(meta_all)

    user_real_all = np.array(final_result['user_score']).reshape(-1, 1)
    model.fit(user_real_all, global_real_all)
    user_all = (f"coefficients of determination for user score "
                f"before filtering: "
                f"{model.score(user_real_all, global_real_all)}")
    print(user_all)

    # filter the data since it looks better about how the points are
    # distributed in the plot and how the regression line shows.
    filter_global_small = (final_result['Global_Sales'] >= 0.5)
    filter_global_big = (final_result['Global_Sales'] <= 10)
    data = final_result[filter_global_small & filter_global_big]
    meta_real_filter = np.array(data['meta_score']).reshape(-1, 1)
    global_real_filter = np.array(data['Global_Sales'])
    model.fit(meta_real_filter, global_real_filter)
    meta_filter = (f"coefficients of determination for meta score "
                   f"after filtering: "
                   f"{model.score(meta_real_filter, global_real_filter)}")
    print(meta_filter)

    user_real_filter = np.array(data['user_score']).reshape(-1, 1)
    model.fit(user_real_filter, global_real_filter)
    user_filter = (f"coefficients of determination for user score "
                   f"after filtering: "
                   f"{model.score(user_real_filter, global_real_filter)}")
    print(user_filter)

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2,
                                                 ncols=2, figsize=(18, 10))
    sns.regplot(data=final_result, x='meta_score', y='Global_Sales', ax=ax1)
    ax1.set_xlabel('Meta Score')
    ax1.set_ylabel('Total Worldwide Sales')
    ax1.title.set_text("Relationship Between Meta Score and Global Sales")

    sns.regplot(data=final_result, x='user_score', y='Global_Sales', ax=ax2)
    ax2.set_xlabel('User Score')
    ax2.set_ylabel('Total Worldwide Sales')
    ax2.title.set_text("Relationship Between Users Score and Global Sales")

    sns.regplot(data=data, x='meta_score', y='Global_Sales', ax=ax3)
    ax3.set_xlabel('Meta Score')
    ax3.set_ylabel('Total Worldwide Sales')
    ax3.title.set_text("Relationship Between Meta Score and Global Sales")

    sns.regplot(data=data, x='user_score', y='Global_Sales', ax=ax4)
    ax4.set_xlabel('User Score')
    ax4.set_ylabel('Total Worldwide Sales')
    ax4.title.set_text("Relationship Between User Score and Global Sales")

    fig.tight_layout()
    plt.savefig('regression analysis.png')
    plt.show()


# Question 5: line chart
def line_plot_change(final_result: pd.DataFrame) -> None:
    """
    Takes in the merged dataset, creates the line chart
    of the top 3 genres' changement over 1996 to 2015,
    and returns nothing.
    """
    # the reason we use these three genre is because
    # these are the top three genre we observed from pie chart
    role = final_result['Genre'] == 'Role-Playing'
    shooter = final_result['Genre'] == 'Shooter'
    action = final_result['Genre'] == 'Action'

    # filter year since there are no data for some genre before 1990
    year = (final_result['Year'] >= 1996) & (final_result['Year'] <= 2015)
    line_data = final_result[(role | shooter | action) & year]
    sns.relplot(x='Year', y='Global_Sales', data=line_data, kind='line',
                hue='Genre', ci=None, aspect=18/10)
    plt.xticks([i for i in range(1996, 2016)], rotation=30)
    plt.title('Players\' Preferences in Top 5 Genres\' Change Over Time')
    plt.ylabel('Total worldwide sales')
    plt.savefig('line_plot_change.png', bbox_inches='tight')
    plt.show()


def main():
    df = get_df()
    final_result = get_final_result()
    pie_chart(final_result)
    histogram_plot(final_result)
    print(f"Test set mean squared error for main analysis using df: "
          f"{fit_and_predict_sales(df, 32, 150, 32)}")
    print(f"Test set mean squared error for main analysis using final_result: "
          f"{fit_and_predict_final_result(final_result, 64, 80, 8)}")
    regression_analysis(final_result)
    line_plot_change(final_result)


if __name__ == '__main__':
    main()