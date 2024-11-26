import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from lightgbm import LGBMRegressor
from preprocessing import DataPreprocessor


data_path = r"..\resale-flat-price-prediction\data\resale_flat_price_jan2017-nov2024.csv"
df = pd.read_csv(data_path)

# Initialize and Preprocess
preprocessor = DataPreprocessor()
df['remaining_lease'] = df['remaining_lease'].apply(preprocessor.process_remaining_lease)
df = preprocessor.process_month(df)
df = preprocessor.process_resale_price(df)
df = preprocessor.optimize_data(df)

#only few features that used
categorical_columns = ['town', 'flat_type', 'flat_model', 'storey_range']
numerical_columns = ['floor_area_sqm', 'remaining_lease','month_sin', 'month_cos', 'month_since_start', 'resale_price_log']

# split
df_full_train, df_test = train_test_split(df[categorical_columns+numerical_columns], test_size=0.2, random_state=42)

#split to train, test, validation
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = (df_train.resale_price_log.values)
y_test = (df_test.resale_price_log.values)
y_val = (df_val.resale_price_log.values)

del df_train['resale_price_log']
del df_test['resale_price_log']
del df_val['resale_price_log']

# model
X_train_dict = df_train.to_dict(orient='records')
X_val_dict = df_val.to_dict(orient='records')
X_test_dict = df_test.to_dict(orient='records')

# Fit DictVectorizer only on the training data
dv = DictVectorizer()
X_train = dv.fit_transform(X_train_dict)

#transform
X_val= dv.transform(X_val_dict)
X_test = dv.transform(X_test_dict)

lgb_reg = LGBMRegressor(random_state=42,
                            learning_rate=0.3,
                            max_depth=10,
                            n_jobs=-1)

# Fit the model on train + validation data
lgb_reg.fit(X_train, y_train)

# Evaluate on the validation set
val_predictions = lgb_reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print("Validation RMSE:", val_rmse)

# Final evaluation on the test set
test_predictions = lgb_reg.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print("Test RMSE:", test_rmse)

with open(r"..\resale-flat-price-prediction\models\model.bin", "wb") as file_out:
    pickle.dump(lgb_reg, file_out)

with open(r"..\resale-flat-price-prediction\models\dv.bin", "wb") as file_out:
    pickle.dump(dv, file_out)
