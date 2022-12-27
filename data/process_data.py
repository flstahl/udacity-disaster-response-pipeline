import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    INPUT:
    messages_filepath - path to messages csv file
    categories_filepath - path to categories csv file
    
    OUTPUT:
    df - Merged data
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    
    return df


def clean_data(df):
    """
    INPUT:
    df - Merged data
    
    OUTPUT:
    df2 - Cleaned data
    """
    
    #split categories into 36 individual columns
    categories = df['categories'].str.split(pat = ';', expand=True)
    
    #assert column names
    row = categories.iloc[0,]
    category_colnames = row.apply(lambda x: x.split('-')[0]).values
    categories.columns = category_colnames
    
    #extract only 0 and 1 from each categories cell
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #drop the original nested/list-like categories column
    df1 = df.drop('categories', axis=1)
    
    #create the final dataframe by joining the messages with the 36 categories columns
    df2 = pd.concat([df1, categories], sort = False, axis=1)
    
    #remove duplicate entries/samples
    df2.drop_duplicates(inplace=True)
    
    
    #upon inspection we noticed that in column 'related' there are not only 0 and 1 but also other values greater than 1. That's why we filter those out and drop the corresponding rows
    df2 = df2.loc[df2['related'] < 2]
    
    return df2

def save_data(df, database_filename):
    """
    INPUT:
    df - cleaned data
    database_filename - database filename for sqlite database with (.db) file type
    
    OUTPUT:
    None - save cleaned data into sqlite database
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse_table', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()