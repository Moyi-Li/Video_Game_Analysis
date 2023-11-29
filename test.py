"""
CSE 163 Group Project
Author: Moyi Li; JingjingDong; Zhikai Li
This program contains testing functions for each research question in main.py.
The main dataset used for testing all research question except question 3
is the small_ds_test.csv, which is a manually created table for
checking the functionality. This table does not have any realistic meaning.
Specifically, the testing for question 3 machine learning part
used the method we've learned in CSE163 lessons
and a test method including different regions
to compare and tests the result.
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from data_cleaning import get_test, get_df, get_final_result

sns.set()


# Quetsion 1:
def test_pie_chart(test: pd.DataFrame) -> None:
    """
    Using the test dataset,
    To check if the function pie_chart run properly.
    """
    genre_counts = test.groupby('Genre')['Name'].count()
    fig = go.Figure(data=[go.Pie(labels=genre_counts.index,
                    values=genre_counts)])
    fig.update_layout(title='Testing Genre Distribution',
                      xaxis_title='Genre', yaxis_title='Name Count')
    fig.show()


# Question 2:
def test_histogram_plot(test: pd.DataFrame) -> None:
    """
    Using the test dataset,
    To check if the function histogram_plot run properly.
    """
    # by checking the dataframe top_publishers,
    # we found out the top 5 publishers
    five_publisher = ['Activision', 'Electronic Arts',
                      'Nintendo', 'Sega', 'Ubisoft']

    top_filtered = test[test['Publisher'].isin(five_publisher)]
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

    plt.title("Testing Region Sales for Top 5 Publishers")
    plt.xlabel('Sales Percentage of Each Region')
    plt.ylabel('Top 5 publishers')
    plt.savefig('test_histogram_plot.png', bbox_inches='tight')
    plt.show()


# Question 3:
def test_fit_and_predict_sklearn(df: pd.DataFrame) -> float:
    """
    This function is the testing function for comparing results. It will
    return a test mean square error of the prediction of Global_Sales with
    using the regression model from sklearn the given data set and three
    integers that determine the number of neurons, the times to iterate
    over the training set, and the number of samples that is used each time.
    """
    df_filtered = df[['Name', 'Platform', 'Year', 'Genre',
                      'Publisher', 'Global_Sales', 'meta_score', 'user_score']]
    df_filtered = df_filtered.dropna()
    features = df_filtered.loc[:, df_filtered.columns != 'Global_Sales']
    features = pd.get_dummies(features)
    labels = df_filtered['Global_Sales']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)
    test_predictions = model.predict(features_test)
    return mean_squared_error(labels_test, test_predictions)


def test_fit_and_predict_sales(test: pd.DataFrame, hid_unit: int,
                               epoch_size: int, batch_size: int) -> float:
    """
    This function is the testing function for our training model, it will
    return a test mean square error of the prediction of Global_Sales with
    the given data set and three integers that determine the number of neurons,
    the times to iterate over the training set, and the number of samples
    that is used each time.
    """
    df_filtered = test[['Platform', 'Year', 'Genre', 'Publisher',
                        'Global_Sales', 'NA_Sales', 'EU_Sales',
                        'JP_Sales', 'Other_Sales']]
    df_filtered = df_filtered.dropna()
    features = df_filtered.loc[:, df_filtered.columns != 'Global_Sales']
    features = pd.get_dummies(features)
    labels = df_filtered['Global_Sales']

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = Sequential()
    model.add(Dense(units=hid_unit, input_dim=features_train.shape[1],
              activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model.fit(features_train, labels_train, epochs=epoch_size,
              batch_size=batch_size, verbose=2,
              validation_data=(features_test, labels_test))
    labels_pred = model.predict(features_test)
    output = mean_squared_error(labels_test, labels_pred)
    return output


# Question 4:
def test_regression_analysis(test: pd.DataFrame) -> None:
    """
    Using the test dataset,
    To check if the function regression_analysis run properly.
    """
    model = LinearRegression()
    meta_test = np.array(test['meta_score']).reshape(-1, 1)
    global_test = np.array(test['Global_Sales'])
    model.fit(meta_test, global_test)
    meta_test = f"meta test: {model.score(meta_test, global_test)}"
    print(meta_test)

    user_test = np.array(test['user_score']).reshape(-1, 1)
    model.fit(user_test, global_test)
    user_test = f"user test: {model.score(user_test, global_test)}"
    print(user_test)

    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    sns.regplot(data=test, x='meta_score', y='NEW_Global_Sales', ax=ax1)
    ax1.set_xlabel('Meta Score')
    ax1.set_ylabel('Total Worldwide Sales')
    ax1.title.set_text("Test for Meta score")

    sns.regplot(data=test, x='user_score', y='NEW_Global_Sales', ax=ax2)
    ax2.set_xlabel('User Score')
    ax2.set_ylabel('Total Worldwide Sales')
    ax2.title.set_text("Test for User score")

    fig.tight_layout()
    plt.savefig('test_regression_analysis.png')
    plt.show()


# Question 5:
def test_line_plot_change(test: pd.DataFrame) -> None:
    """
    Using the test dataset,
    To check if the function fit_and_predict_sales run properly.
    """
    role = test['Genre'] == 'Role-Playing'
    shooter = test['Genre'] == 'Shooter'
    action = test['Genre'] == 'Action'

    line_data = test[(role | shooter | action)]
    sns.relplot(x='Year', y='Global_Sales', data=line_data, kind='line',
                hue='Genre', errorbar=None, aspect=18/10)
    plt.xticks([i for i in range(1996, 2016)], rotation=30)
    plt.title('Testing Plot')
    plt.ylabel('Total worldwide sales')
    plt.savefig('test_line_plot.png', bbox_inches='tight')
    plt.show()


def main():
    test = get_test()
    df = get_df()
    final_result = get_final_result()
    test_pie_chart(test)
    test_histogram_plot(test)
    print(f"Test set mean squared error with sales column: "
          f"{test_fit_and_predict_sales(df, 32, 150, 32)}")
    print(f"Test set mean squared error with sklearn: "
          f"{test_fit_and_predict_sklearn(final_result)}")
    test_line_plot_change(test)
    test_regression_analysis(test)


if __name__ == '__main__':
    main()