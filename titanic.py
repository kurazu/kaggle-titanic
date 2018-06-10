
import io
import os
import os.path
import multiprocessing
import random
import uuid
import functools
import pickle

import numpy as np

import pandas

import sklearn.preprocessing
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer

from sklearn_pandas import DataFrameMapper

from keras import backend as K
from keras.models import Sequential
from keras.layers import (
    Dense, InputLayer, BatchNormalization, LeakyReLU, Dropout
)
from keras.metrics import binary_accuracy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.wrappers.scikit_learn import KerasClassifier


class SessionCleanCallback(Callback):

    def __init__(self):
        super().__init__()
        self.params = None
        with io.open('out.log', 'w', encoding='utf-8') as f:
            f.write(f'Main {os.getpid()}\n')
            f.flush()
        self.counter = 0

    def write(self, text):
        with io.open('out.log', 'a', encoding='utf-8') as f:
            f.write(f'{os.getpid()}: ')
            f.write(text)
            f.write('\n')
            f.flush()

    def set_model(self, model):
        self.write(f'set model {self.model == model}')
        super().set_model(model)
        # K.clear_session()

    def set_params(self, params):
        self.write(f'set params {self.params == params}')
        super().set_params(params)

    def on_train_begin(self, logs=None):
        self.write('begin')

    def on_train_end(self, logs=None):
        self.counter += 1
        self.write(f'end {self.counter}')
        if self.counter % 3 == 0:
            K.clear_session()


def here_path(*components):
    return os.path.join(os.path.dirname(__file__), *components)


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
PATIENCE = 30
MODEL_FILE = 'weights.best.hdf5'
MODEL_PATH = here_path(MODEL_FILE)
VERBOSE = 0

# BATCH_SIZE = 128
# LEARNING_RATE = 0.001
# VALIDATION_SPLIT = 0.2
# LAYERS = [128, 32]
# CONTROL_VAR = 'binary_accuracy'

ROUNDS = 100
PARAMS = {
    'dropout_rate': [0.0, 0.25, 0.5],
    'batch_size': [16, 32, 64, 128, 256],
    'learning_rate': [
        0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01
    ],
    'validation_split': [0.2, 0.3],
    'layers': [
        (1024, 512, 128, 32),
        (1024, 512, 128, 16),
        (1024, 256, 64),
        (1024, 256, 32),
        (512, 128, 32),
        (512, 64, 16),
        (256, 64),
        (128, 32),
        (64, 16),
        (128, ),
        (64, ),
        (32, )
    ]
}
INPUT_SHAPE = ()


def get_model(layers=(512, 128, 32), learning_rate=0.0001, dropout_rate=0.0):
    K.clear_session()
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))

    for units in layers:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=learning_rate),
        metrics=['binary_accuracy']
    )

    return model


def get_callback():
    stopping = EarlyStopping(
        monitor='val_binary_accuracy',
        min_delta=0,
        patience=PATIENCE,
        verbose=VERBOSE,
        mode='max'
    )
    return stopping


def get_callbacks():
    checkpointer = ModelCheckpoint(
        MODEL_FILE, monitor='val_binary_accuracy',
        verbose=VERBOSE, save_best_only=True, mode='max'
    )

    return [checkpointer, get_callback()]


def train(
    train_X, train_y,
    batch_size, learning_rate, validation_split, layers, dropout_rate
):
    model = get_model(layers, learning_rate, dropout_rate)
    callbacks = get_callbacks()

    history = model.fit(
        train_X, train_y,
        validation_split=validation_split,
        shuffle=True,
        epochs=EPOCHS, batch_size=batch_size,
        callbacks=callbacks,
        verbose=VERBOSE
    )
    # Load the best set of weights
    model.load_weights(MODEL_FILE)

    return model, np.max(history.history['val_binary_accuracy'])


def evaluate(train_X, train_y, model):
    scores = model.evaluate(train_X, train_y, verbose=VERBOSE)
    scores_dict = {
        label: score for label, score in zip(model.metrics_names, scores)
    }
    return scores_dict['binary_accuracy']


def hyperparam_search(train_X, train_y, params):
    model = KerasClassifier(
        build_fn=get_model, epochs=EPOCHS, batch_size=128, verbose=VERBOSE
    )
    # grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, verbose=1)
    grid = RandomizedSearchCV(
        scoring='accuracy',
        estimator=model, param_distributions=params,
        n_jobs=-1, verbose=2, n_iter=ROUNDS
    )
    grid_result = grid.fit(train_X, train_y, callbacks=[get_callback()])
    return grid_result.best_score_, grid_result.best_params_


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

    global INPUT_SHAPE
    _, *INPUT_SHAPE = train_X.shape

    # best_accuracy, best_params = find_best_model(
    #     train_X, train_y, ROUNDS, PARAMS
    # )
    best_accuracy, best_params = hyperparam_search(
        train_X, train_y, PARAMS
    )
    print('Best accuracy', best_accuracy)
    print('Best params', best_params)

    # best_params = {
    #     'batch_size': 128,
    #     'learning_rate': 0.0001,
    #     'validation_split': 0.2,
    #     'layers': (512, 64, 16),
    #     'control_var': 'val_binary_accuracy'
    # }

    model, accuracy = train(train_X, train_y, **best_params)
    print('Retrained model accuracy', accuracy)

    accuracy_on_whole_set = evaluate(train_X, train_y, model)
    print('Retrained model accuracy on whole set', accuracy_on_whole_set)

    predict_y = model.predict(predict_X, batch_size=32)

    predictions = (predict_y.reshape((-1, )) >= 0.5).astype('int64')

    output_frame = pandas.DataFrame({
        'Survived': predictions
    }, index=ids)
    output_frame.to_csv(here_path('submission.csv'))


if __name__ == '__main__':
    main()
