import os.path

import numpy as np

import pandas

import sklearn.preprocessing
from sklearn.model_selection import train_test_split

from sklearn_pandas import DataFrameMapper

from keras.models import Sequential
from keras.layers import Dense, InputLayer, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def here_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def kaggle_path(competition, filename):
    return os.path.expanduser(
        os.path.join('~', '.kaggle', 'competitions', competition, filename)
    )


def get_mapper():
    return DataFrameMapper([
        ('Pclass', sklearn.preprocessing.LabelBinarizer()),
        ('Sex', sklearn.preprocessing.LabelBinarizer()),
        (['Age'], sklearn.preprocessing.StandardScaler()),
        (['SibSp'], sklearn.preprocessing.StandardScaler()),
        (['Parch'], sklearn.preprocessing.StandardScaler()),
        (['Fare'], sklearn.preprocessing.StandardScaler()),
        ('Embarked', sklearn.preprocessing.LabelBinarizer()),
    ])


def analyze(data_frame):
    print('*' * 80)
    all_rows = len(data_frame)
    for column_name in data_frame.columns:
        column = data_frame[column_name]
        missing_values = all_rows - column.count()
        print(
            'Column', column_name,
            'type', column.dtype,
            'missing values', missing_values
        )


def cleanup_nan(data_frame):
    data_frame.Embarked.fillna(
        data_frame.Embarked.value_counts().idxmax(), inplace=True
    )
    data_frame.Age.fillna(data_frame.Age.mean(), inplace=True)
    data_frame.Fare.fillna(data_frame.Fare.mean(), inplace=True)


def get_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape[1:]))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=LEARNING_RATE),
        metrics=['binary_accuracy']
    )

    return model


EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def main():
    train_data_frame = pandas.read_csv(kaggle_path('titanic', 'train.csv'))
    test_data_frame = pandas.read_csv(kaggle_path('titanic', 'test.csv'))

    analyze(train_data_frame)
    analyze(test_data_frame)

    cleanup_nan(train_data_frame)
    cleanup_nan(test_data_frame)

    analyze(train_data_frame)
    analyze(test_data_frame)

    mapper = get_mapper()
    mapper.fit(train_data_frame)

    train_X = mapper.transform(train_data_frame)
    train_y = train_data_frame.as_matrix(columns=['Survived'])
    predict_X = mapper.transform(test_data_frame)
    ids = test_data_frame.PassengerId

    train_features, valid_features, train_labels, valid_labels = \
        train_test_split(train_X, train_y, test_size=0.2)

    print('Train X', train_features.shape)
    print('Train y', train_labels.shape)
    print('Validation X', valid_features.shape)
    print('Validation y', valid_labels.shape)
    print('Predict X', predict_X.shape)

    model = get_model(train_X.shape)

    stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto'
    )

    model.fit(
        train_features, train_labels,
        shuffle=True,
        validation_data=(valid_features, valid_labels),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[stopping]
    )

    predict_y = model.predict(predict_X, batch_size=BATCH_SIZE)

    predictions = (predict_y.reshape((-1, )) >= 0.5).astype('int64')

    output_frame = pandas.DataFrame({
        'Survived': predictions
    }, index=ids)
    output_frame.to_csv(here_path('submission.csv'))


if __name__ == '__main__':
    main()
