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
PATIENCE = 30
MODEL_FILE = 'weights.best.hdf5'
MODEL_PATH = here_path(MODEL_FILE)

# BATCH_SIZE = 128
# LEARNING_RATE = 0.001
# VALIDATION_SPLIT = 0.2
# LAYERS = [128, 32]
# CONTROL_VAR = 'binary_accuracy'

ROUNDS = 10
PARAMS = {
    'batch_size': [16, 32, 64, 128, 256],
    'learning_rate': [
        0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01
    ],
    'validation_split': [0.2, 0.3, 0.4, 0.5],
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
    ],
    'control_var': [
        'val_loss', 'val_accuracy', 'val_binary_accuracy'
    ]
}


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

    checkpointer = ModelCheckpoint(
        MODEL_PATH, monitor=control_var,
        verbose=1, save_best_only=True, mode='max'
    )

    return [stopping, checkpointer]


def train(
    train_X, train_y,
    batch_size, learning_rate, validation_split, layers, control_var
):
    model = get_model(train_X.shape, layers, learning_rate)
    callbacks = get_callbacks(control_var)

    model.fit(
        train_X, train_y,
        validation_split=validation_split,
        shuffle=True,
        epochs=EPOCHS, batch_size=batch_size,
        callbacks=callbacks
    )

    model.load_weights(MODEL_PATH)

    scores = model.evaluate(train_X, train_y, verbose=1)
    scores_dict = {
        label: score for label, score in zip(model.metrics_names, scores)
    }
    return model, scores_dict['binary_accuracy']


def find_best_model(train_X, train_y, rounds, params):
    best_model = None
    best_accuracy = 0.0
    best_params = {
        param: values[0] for param, values in params.items()
    }

    for round_idx in range(rounds):
        for param, values in params.items():
            print('Round', round_idx + 1, 'querying param', param)
            for value in values:
                candidate_params = dict(best_params)
                candidate_params[param] = value
                model, accuracy = train(
                    train_X, train_y, **candidate_params
                )
                if accuracy > best_accuracy:
                    print(
                        'Round', round_idx + 1,
                        'found better param', param, ':',
                        best_params[param], '=>', value,
                        '(', best_accuracy, '=>', accuracy, ')'
                    )
                    best_accuracy = accuracy
                    best_model = model
                    best_params = candidate_params

    return best_model, best_accuracy, best_params


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

    model, accuracy, params = find_best_model(train_X, train_y, ROUNDS, PARAMS)
    model.save_weights(MODEL_PATH)

    print('Best accuracy', accuracy)
    print('Best params', params)

    predict_y = model.predict(predict_X, batch_size=32)

    predictions = (predict_y.reshape((-1, )) >= 0.5).astype('int64')

    output_frame = pandas.DataFrame({
        'Survived': predictions
    }, index=ids)
    output_frame.to_csv(here_path('submission.csv'))


if __name__ == '__main__':
    main()
