import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger','maxent_ne_chunker','words'])

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from operator import itemgetter
import joblib
import re

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.neighbors import KNeighborsClassifier




def load_data(database_filepath):
    '''
    Loads data from database, feature and target variables X and Y are defined
    - Input: database filepath
    - Output: X, Y and category names
    '''
    #load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Clean_Disaster_Data', engine)  

    #define X,Y and category_names
    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns
    
    return X,Y,category_names


def tokenize(text):
    '''
    Tokenization function to process text data using word tokenization, stemming and lemmatizer
    - Imput: Complete text as string
    - Output: Clean tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#    # normalizing all the text
#    text = text.lower()
#
#    # removing extra characters
#    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
#
#    # tokenizing all the sentences
#    words = word_tokenize(text)
#
#    # removing stopwords
#    words = [w for w in words if w not in stopwords.words("english")]
#
#    # Reduce words to their stems
#    stemmed = [PorterStemmer().stem(w) for w in words]
#
#    # Lemmatize verbs by specifying pos
#    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in stemmed]
#
#    return lemmed

    

def build_model(pipeline, parameters):
    '''
    Trains pipeline
    - Split data into train and test sets
    - Train pipeline
    Input: Machine pipeline taking in the `message` column as input and output classification results on the other 36 categories in the dataset and parameters for model hypertuning. 
    Output: Model using best set of parameters
    '''

    # Perform grid search using the pipeline
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Reports the accuracy, precision and recall for each output category of the dataset and an overall model quality.
    - Input: model, X_test, Y_test and list of categories
    - Output: model evaluation for each category stored in Dataframe
    '''
    #Predictions
    Y_pred = model.predict(X_test)

    #Model evaluation
    model_eval = pd.DataFrame(columns=['target_category','accuracy','precision','recall'])

    for col in range(0,len(category_names)):
    
        report = classification_report(Y_test.values[col], Y_pred[col],output_dict=True,zero_division=0.0)
        result_dict = {"target_category":category_names[col],
                    "accuracy":report['accuracy'], 
                    "precision": report['macro avg']['precision'],
                    "recall": report['macro avg']['recall']
                    }
        result = pd.DataFrame([result_dict])
        print(result)
        model_eval = model_eval.append(result)

    print('......\n'
          'Model Quality:\n '
          'Model Accuracy: ',model_eval['accuracy'].mean(),'(+/- ', model_eval['accuracy'].std(),')\n ' 
          'Model Precision: ',model_eval['precision'].mean(),'(+/- ', model_eval['precision'].std(),')\n ' 
          'Model Recall: ',model_eval['recall'].mean(),'(+/- ', model_eval['recall'].std(),')' )

    return model_eval
  

def save_model(model, model_filepath):
    '''
    Exports model as a pickle file
    '''
    #saving best classifier as pickle
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        #define the pipeline with transformers and models
        pipeline =  Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])
        
        parameters = {
                'clf__estimator__n_estimators': [50, 100, 200],
                'clf__estimator__max_depth': [None, 5, 10, 20],
                'clf__estimator__min_samples_split': [2, 3, 4]
            }   
        

        pipeline2 =  Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(KNeighborsClassifier()))
            ])

        parameters2 = {
                'clf__estimator__n_neighbors': list(range(1, 31))
            }  

        print('Building model...')
        model1 = build_model(pipeline, parameters)
        model2 = build_model(pipeline2, parameters2)
        
        print('Training model...')
        model1.fit(X_train, Y_train)
        model2.fit(X_train, Y_train)

        print('Evaluating model...')
        result1 = evaluate_model(model1, X_test, Y_test, category_names)
        result2 = evaluate_model(model2, X_test, Y_test, category_names)

        # determine the best model
        print('Determine best model...')
        if result1['accuracy'].mean() > result2['accuracy'].mean():
            best_model = model1
        elif result1['accuracy'].mean() < result2['accuracy'].mean():
            best_model = model2
        print('Best Model is {}.'.format(best_model))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()