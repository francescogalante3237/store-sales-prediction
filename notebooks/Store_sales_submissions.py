import pandas as pd
from pathlib import Path
import numpy as np

from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def autoregressive_forecast(model, df_model_test, features, lags=(1, 7, 14, 28)):
    """
    Perform day-by-day autoregressive forecasting.
    
    Parameters
    ----------
    model : trained model
        Model trained on log-transformed target (log1p(sales)).
    df_model_test : pd.DataFrame
        Test set containing lag features and other predictors.
    features : list
        List of feature column names used for prediction.
    lags : tuple of int
        Lag periods to update (default: (1, 7, 14, 28)).

    Returns
    -------
    pd.DataFrame
        Copy of df_model_test with columns:
        - 'pred': predicted log-sales
        - 'sales': predicted sales in original scale
        - updated lag features
    """

    df_pred = df_model_test.copy()
    df_pred['pred'] = np.nan
    test_dates = sorted(df_pred['date'].unique())
    key_cols = ['store_nbr', 'family']

    for date in test_dates:
        mask_day = df_pred['date'] == date
        X_day = df_pred.loc[mask_day, features].copy()

        preds = model.predict(X_day)
        df_pred.loc[mask_day, 'pred'] = preds

        # update lag features for future days
        for lag in lags:
            target_date = date + pd.Timedelta(days=lag)
            if target_date in df_pred['date'].values:
                mask_future = df_pred['date'] == target_date

                df_future = df_pred.loc[mask_future, key_cols]
                df_today = df_pred.loc[mask_day, key_cols].copy()
                df_today['pred_val'] = preds

                merged = pd.merge(df_future, df_today, on=key_cols, how='left')
                df_pred.loc[mask_future, f'sales_lag_{lag}'] = np.expm1(merged['pred_val'].values)

    df_pred['sales'] = np.expm1(df_pred['pred'])
    return df_pred




################# IMPORT DATA

PROJECT_ROOT = Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()
RAW_PATH = PROJECT_ROOT / "data" / "raw"

files = {
    "train": "train.csv",
    "test": "test.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv"
}

df_train, df_test, df_oil, df_holiday = [pd.read_csv(RAW_PATH / fname) for fname in files.values()]

################# FEATURE ENGINEERING

# Add a column 'sales' to the test data frame (the target) filled with NaN. Now we can stack train and test together and easily compute the features for both
df_test['sales'] = np.nan

# Fix datetime format
df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

# Save the first date in the test set, so we will know how to split back
first_test_date = df_test['date'].min() # we will use it later for splitting train and test

# Concatenate the two dataset 
df = pd.concat([df_train, df_test], ignore_index=True, sort=False)

# Change the type of the 'onpromotion' feature
df['onpromotion'] = df['onpromotion'].astype('float32') # important for speeding up training


# sort the df first for store number, than family product than date: all time series will be stacked
df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

# define lags for lagged features
lags = [1, 7, 14, 28]

# create a feature for each lag
for lag in lags:
    df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)
    df[f'onpromo_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['onpromotion'].shift(lag)

# we add calendar features
df['dow'] = df['date'].dt.dayofweek # Day of the week (e.g. takes into account weekends)
df['dom'] = df['date'].dt.day # Number of the month (e.g. salaries are given twice a month)
df['month'] = df['date'].dt.month # Learns relevant month e.g. december for christmas
df[['dow','dom','month']] = df[['dow','dom','month']].astype('int16') 


# encode categorical
# IMPORTANT: train and validation as well as test contain the same categories so encoding now (before splitting) does not leak information
# We create a new feature containing encoded product family (cat 1, cat, 2 etc)
enc = OrdinalEncoder()
df['family_enc'] = enc.fit_transform(df[['family']]).astype('int16')

# To make the model learn a decay effect after the earthquake event of April 16, 2016
# Adds a new column containing the number of days after the event
eq_day = '2016-04-16'
df['days_since_eq'] = (df['date'] - pd.Timestamp(eq_day)).dt.days
# reset to nan days before the earthquake to prevent a countdown and leak info
df.loc[df['days_since_eq'] < 0, 'days_since_eq'] = np.nan

# OIL PRICE

# Fix datetime format
df_oil['date'] = pd.to_datetime(df_oil['date'])
# Change the column name to a more readable one
df_oil = df_oil.rename(columns={'dcoilwtico': 'oil_price'})

# There are several days that:
#  - Are not listed in df_oil (holes)
#  - Are listed but the oil price is NaN

# We consider all the dates appearing in the original dataset 
all_dates = pd.DataFrame({'date': df['date'].dropna().unique()})

# and we add new rows to df_oil with the missing dates, setting price to NaN
df_oil = pd.merge(all_dates, df_oil, on='date', how='left').sort_values('date').reset_index(drop=True)
 
# Propagate forward the last observed price (the first value was NaN so it remains)
df_oil['oil_price'] = df_oil['oil_price'].ffill()

# We merge with the original dataset: now every date has the last observed oil price (except the first one)
df = pd.merge(df, df_oil, on='date', how='left')

# lag_7 oil price feature
df['oil_price_lag_7'] = df['oil_price'].shift(7)


# choose features
lag_cols = [c for c in df.columns if c.startswith(('sales_lag_', 'onpromo_lag_','oil_price_lag_'))]
base_cols = ['onpromotion','store_nbr','family_enc','dow','dom','month','days_since_eq','oil_price']

features = lag_cols  + base_cols

####################################### TRAIN TEST SPLIT
# we make a copy of the dataset
df_model = df.copy()

# Get train and test with engineered features
df_model_train = df_model[df_model['date'] < first_test_date].copy() #  -> this goes into the training set
df_model_test  = df_model[df_model['date'] >= first_test_date].copy() # -> this goes into the test set

# Log target for RMSLE: we add a new column to the train set with the transformed target
target = 'target'
df_model_train[target] = np.log1p(df_model_train['sales']) # We will use this transformed version of 'sales' as target

X_tr = df_model_train[features] # features for train and validation
y_tr = df_model_train[target]   # target for train and validation

# There is at least one nan feature for the first 28 days. We remove those features.
# However we must keep the nan's in the 'days_since_eq' column
allowed_nan = ['days_since_eq']
mask_tr = X_tr.drop(columns=allowed_nan, errors='ignore').notna().all(axis=1)
X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]

############## MODEL ##################


model = XGBRegressor(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    tree_method='hist',
    random_state=42,
    eval_metric='rmse'
)

model.fit(X_tr, y_tr, 
          eval_set=[(X_tr, y_tr)],   
          verbose=10)  


df_pred_va = autoregressive_forecast(model, df_model_test, features, lags=(1, 7, 14, 28))
pred_test= df_pred_va.loc[df_model_test.index, 'sales'].values

submission = pd.DataFrame({
    "id": df_model_test["id"],
    "sales": pred_test
})
submission = submission.sort_values('id').reset_index(drop=True)


submission.to_csv("submission.csv", index=False)
print("Saved:", submission.shape, "-> submission.csv")