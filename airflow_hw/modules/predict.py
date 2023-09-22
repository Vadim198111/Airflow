from datetime import datetime
import glob
import json
import os

import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    mod = sorted(os.listdir(f'{path}/data/models/'))[-1]
    with open(f'{path}/data/models/{mod}', 'rb') as file:
        model = dill.load(file)

    df = pd.DataFrame(columns=['car_id', 'pred'])
    for test_model in glob.glob(f'{path}/data/test/*.json'):
        with open(test_model) as file:
            form = json.load(file)

            df_test = pd.DataFrame.from_dict([form])
            pred = model.predict(df_test)
            test_dict = {'car_id': df_test.id, 'pred': pred}
            df_predict = pd.DataFrame(test_dict)
            predicted = pd.concat([df, df_predict], axis=0)
            df = predicted
    df.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()


