import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

TEST_SIZE = 0.25
RANDOM_STATE = 1234
PATH_DATA = '..\row\online_shoppers_intention_eda.csv'
categorical_features = ['Month','VisitorType']
numeric_features =  ['Administrative_Duration','Informational_Duration','ProductRelated_Duration',
                    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
                    'OperatingSystems', 'Region', 'TrafficType']


if __name__=='__main__':

    df = pd.read_csv(PATH_DATA)

    # разделим данные на целевую переменную и матрицу объект-признак с учетом дополнительных категориальных признаков
    y = df['Revenue'] # целевая переменная
    X = df[['Administrative_Duration','Informational_Duration','ProductRelated_Duration',
                    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
                    'OperatingSystems', 'Region', 'TrafficType',
                    'Month','VisitorType']] # матрица объект-признак

    #разделим выборку на тренировочную и тестовую
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = RANDOM_STATE)
    y_test.to_csv(r'.\models\y_test.csv',index = False)

    #создадим трансформер кодирующий числовые и категориальные переменные 
    # + добавим масштабирование числовых признаков
    ct = ColumnTransformer([
                           ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
                           ('scaling', MinMaxScaler(), numeric_features)
                           ])

    #для использования ClassifierExplainer приведем X_test_transformed к типу dataframe 
    X_test_transformed = ct.fit_transform(X_test)

    new_features = list(ct.named_transformers_['ohe'].get_feature_names_out())
    new_features.extend(numeric_features)

    X_test_transformed = pd.DataFrame(X_test_transformed, columns=new_features)
    X_test_transformed.to_csv(r'.\models\X_test_transformed.csv',index = False)

    #итоговый pipeline
    pipe = Pipeline([
                    ('transformer', ct), # преобразование данных    
                    ('model', RandomForestClassifier(random_state = 12345, class_weight = {0: 0.2, 1: 1})) # обучение модели
                    ])

    #подберем параметры
    params = {'model__n_estimators' : np.arange(100, 300, 50),
              'model__criterion' : ['gini', 'entropy', 'log_loss'],
              'model__max_depth': np.arange(2, 10, 2)}

    gs = GridSearchCV(pipe, params, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)
    gs.fit(X_train, y_train)

    #сохраним полученную модель
    joblib.dump(gs.best_estimator_,r'.\models\best_model.pkl')




