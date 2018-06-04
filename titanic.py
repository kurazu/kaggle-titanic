import os.path

import pandas
import sklearn.preprocessing
from sklearn_pandas import DataFrameMapper


def kaggle_path(competition, filename):
    return os.path.expanduser(
        os.path.join('~', '.kaggle', 'competitions', competition, filename)
    )


# mapper = DataFrameMapper([
#     (['age'], sklearn.preprocessing.StandardScaler()), # single transformation
#     ('sex', sklearn.preprocessing.LabelBinarizer()), # single transformation
#     ('native_country', [ # multiple transformations
#         sklearn.preprocessing.FunctionTransformer(
#             native_country_generalize, validate=False
#         ),
#         sklearn.preprocessing.LabelBinarizer()
#     ]),
#     ...
# ])

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


def main():
    train_data_frame = pandas.read_csv(kaggle_path('titanic', 'train.csv'))
    test_data_frame = pandas.read_csv(kaggle_path('titanic', 'test.csv'))

    analyze(train_data_frame)
    analyze(test_data_frame)


if __name__ == '__main__':
    main()
