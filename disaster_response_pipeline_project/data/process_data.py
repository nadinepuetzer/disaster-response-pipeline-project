import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input: Messages datafile and categories datafile
    Output: Messages and categories dataframe merged together (df)
    '''

    #Load messages dataset
    messages = pd.read_csv(messages_filepath)  

    # load categories dataset
    categories = pd.read_csv(categories_filepath)  
    
    # merge datasets
    df = messages.merge(categories, how='outer', on=['id'])
    
    return df

def clean_data(df):
    '''
    Data Cleaning Steps:
    - Split `categories` into separate category columns.
    - Convert category values to just numbers 0 or 1.
    - Replace `categories` column in `df` with new category columns.
    - Remove duplicates.
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    # set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df=df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1)
    
    # drop duplicates and rows with related==2
    df = df.loc[df["related"] <= 1 ]
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    Data are stored on an sqllite database in the table Clean_Disaster_Data
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Clean_Disaster_Data', engine, index=False, if_exists='replace' )  


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