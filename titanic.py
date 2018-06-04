import os.path

import numpy as np

import pandas

import sklearn.preprocessing
from sklearn.model_selection import train_test_split

from sklearn_pandas import DataFrameMapper

from keras.models import Sequential
from keras.layers import Dense, InputLayer, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


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


EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PATIENCE = 20
VALIDATION_SPLIT = 0.2
MODEL_FILE = 'weights.best.hdf5'
LAYERS = [128, 32]
CONTROL_VAR = 'binary_accuracy'


def get_model(input_shape, layers, learning_rate):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape[1:]))

    for units in layers:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=learning_rate),
        metrics=['binary_accuracy']
    )

    return model


def get_callbacks(control_var):
    stopping = EarlyStopping(
        monitor=control_var,
        min_delta=0,
        patience=PATIENCE,
        verbose=1,
        mode='auto'
    )

    weights_path = here_path(MODEL_FILE)

    checkpointer = ModelCheckpoint(
        weights_path, monitor=control_var,
        verbose=1, save_best_only=True, mode='max'
    )

    return [stopping, checkpointer], weights_path


def train(
    train_X, train_y,
    batch_size, learning_rate, validation_split, layers, control_var
):
    model = get_model(train_X.shape, layers, learning_rate)
    callbacks, weights_path = get_callbacks(control_var)

    model.fit(
        train_X, train_y,
        validation_split=validation_split,
        shuffle=True,
        epochs=EPOCHS, batch_size=batch_size,
        callbacks=callbacks
    )

    model.load_weights(weights_path)

    scores = model.evaluate(train_X, train_y, verbose=1)
    scores_dict = {
        label: score for label, score in zip(model.metrics_names, scores)
    }
    return model, scores_dict['binary_accuracy']


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

    print('Train X', train_X.shape)
    print('Train y', train_y.shape)
    print('Predict X', predict_X.shape)

    model, accuracy = train(
        train_X, train_y,
        BATCH_SIZE, LEARNING_RATE, VALIDATION_SPLIT, LAYERS, CONTROL_VAR
    )

    print('Accuracy', accuracy)

    predict_y = model.predict(predict_X, batch_size=BATCH_SIZE)

    predictions = (predict_y.reshape((-1, )) >= 0.5).astype('int64')

    output_frame = pandas.DataFrame({
        'Survived': predictions
    }, index=ids)
    output_frame.to_csv(here_path('submission.csv'))


if __name__ == '__main__':
    main()
