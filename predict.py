from datetime import datetime

import json
import os

import dill

import pandas as pd

path = os.environ.get('~/airflow_hw/', '.')


def predict():
    model_path = f'{path}/data/models/cars_pipe_202410081421.pkl'
    test_data_path = f'{path}/data/test'
    predictions_output_path = f'{path}/data/predictions'

    with open(model_path, 'rb') as file:
        model = dill.load(file)

        all_predictions = []
        files = [7310993818, 7313922964, 7315173150, 7316152972, 7316509996]
        for i in files:
            with open (f'{test_data_path}/{i}.json') as json_file:
                form = json.load(json_file)
                df = pd.DataFrame.from_dict([form])
                y = model.predict(df)

                all_predictions.append({
                    'id': form['id'],
                    'prediction': y[0]
                })

                # Преобразуем результаты в DataFrame
        predictions_df = pd.DataFrame(all_predictions)

                # Сохраняем DataFrame в CSV-файл
        pred_filename = f'{predictions_output_path}/pred_{datetime.now().strftime("%Y%m%d%H%M")}'

        predictions_df.to_csv(pred_filename, index=False)

    print(all_predictions)


if __name__ == '__main__':
    predict()








