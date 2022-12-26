import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, auc, classification_report, make_scorer
import nltk
from sqlalchemy import create_engine, inspect
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def load_data(database_filepath):
    """
    INPUT:
    database_filepath - filepath of database with disaster messages
    
    
    OUTPUT:
    X - messages (input variable) 
    Y - categories of the messages (output variable)
    category_names - category name for y
    """
    
    #read the data from previously created database and store as dataframe df
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name='DesasterResponse_table', con = engine)
    
    #split the data into input data X and output data (labels) Y
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    """
    INPUT:
    text - raw text
    
    OUTPUT:
    clean_tokens - tokenized messages
    """
    tokens = word_tokenize(text.lower())
    
    Lemmatizer = WordNetLemmatizer()
    for token in tokens:
        token = Lemmatizer.lemmatize(token).strip()
        
    return tokens   


def build_model():
    """
    INPUT:
    none
    
    OUTPUT:
    pipeline = ML model pipeline 
    """
    pipeline = Pipeline([('Features', Pipeline([
                               ('count_vect', CountVectorizer(tokenizer=tokenize)),
                                ('tfidf', TfidfTransformer())])),
                    ('Classifier', MultiOutputClassifier(estimator=RandomForestClassifier()))])
    
    parameters = {
    'Features__count_vect__max_df': (.5,1),
    'Features__count_vect__max_features': (None, 1000),
    'Features__tfidf__use_idf': (True, False)
    }

    scorer = make_scorer(scoring_metric)
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring = scorer)

    cv.fit(X_train, y_train)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
    model - ML model
    X_test - test messages
    y_test - categories for test messages
    category_names - category name for y
    
    OUTPUT:
    none - print scores (precision, recall, f1-score) for each output category of the dataset.
    """
    Y_pred_test = model.predict(X_test)
    
    for i in np.arange(len(category_names)):
        print('Category: ' + category_names[i])
        print(classification_report(np.array(Y_test.iloc[:,i]), Y_pred_test[:,i]))
    #print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))

def save_model(model, model_filepath):
    """
    INPUT:
    model - ML model
    model_filepath - location to save the model
    
    OUTPUT:
    none
    """
    
    # save model in pickle file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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