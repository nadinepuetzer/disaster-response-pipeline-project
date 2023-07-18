import sys
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger'])

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from operator import itemgetter
import joblib


from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

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
from sklearn.ensemble import GradientBoostingClassifier


def load_data(database_filepath):
    #load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Clean_Disaster_Data', engine)  

    #define X,Y and category_names
    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns
    
    return X,Y,category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


#additional Features
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    

def build_model():
#    #define the pipeline with transformers and models
#    pipeline = Pipeline([
#        ('features', FeatureUnion([
#    
#            ('text_pipeline', Pipeline([
#                ('vect', CountTransformer(tokenizer=tokenize)),
#                ('tfidf', TfidfVectorizer())
#            ])),
#
#            ('starting_verb', StartingVerbExtractor())
#        ])),
#        ('model1', MultiOutputClassifier(RandomForestClassifier())),
#        ('model2', MultiOutputClassifier(KNeighborsClassifier()))
#    ])
#
#    #define parameter grid
#    parameters = {
#            'model1__estimator__n_estimators': [50, 100, 200],
#            'model1__estimator__min_samples_split': [2, 3, 4],
#            'model2__estimator__n_neighbors': list(range(1, 31))
#        }

    # Define the pipeline with transformers and models
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model1', RandomForestClassifier()),
        ('model2', GradientBoostingClassifier())
    ])
    # Define the hyperparameter grid for transformers and models
    param_grid = {
        'tfidf__max_features': [1000, 2000, 3000],
        'model1__n_estimators': [100, 200, 300],
        'model1__max_depth': [None, 5, 10],
        'model2__n_estimators': [100, 200, 300],
        'model2__learning_rate': [0.1, 0.2, 0.3]
    }
    # Perform grid search using the pipeline
    model = GridSearchCV(pipeline, param_grid, cv=3)

    #cross validation using GridSearchCV
    #model = GridSearchCV(pipeline, param_grid=parameters, cv=2, refit=True, n_jobs=-1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    #Predictions
    Y_pred = model.predict(X_test)

    #Model evaluation
    model_eval = pd.DataFrame(columns=['target_category','accuracy','precision','recall'])

    for col in range(0,len(category_names)):
        report = classification_report(Y_test.values[col], Y_pred[col],output_dict=True)
        #print("Category: {}, Accuracy: {:0.2f}, Precision: {:0.2f}, Recall: {:0.2f}".format(category_names[col],report['accuracy'], report['macro avg']['precision'], report['macro avg']['recall']))
        result = pd.DataFrame({"target_category":category_names[col],
                    "accuracy":report['accuracy'], 
                    "precision": report['macro avg']['precision'],
                    "recall": report['macro avg']['recall']
                    })
        model_eval.append(result)
        print('Model Accuracy: ',model_eval['accuracy'].mean(),'(+/- ', model_eval['accuracy'].std(),')/n ' /
              'Model Precision: ',model_eval['precision'].mean(),'(+/- ', model_eval['precision'].std(),')/n ' /
              'Model Recall: ',model_eval['recall'].mean(),'(+/- ', model_eval['recall'].std(),')' )
              


def save_model(model, model_filepath):
    #saving best classifier as pickle
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()