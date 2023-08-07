# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 09:12:12 2023

@author: nehak
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import numpy as np
from plotly.subplot import 
import plotly.graph_objects as go


df_technical = pd.read_csv(
    r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\prices-split-adjusted.csv")
df_fundamental = pd.read_csv(
    r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\fundamentals.csv")
df_fundamental = df_fundamental.drop("Unnamed: 0", axis=1)
df_securities = pd.read_csv(
    r"C:\Neha\kaggle Projects\Git hub\Finance\Dataset\securities.csv")


print(df_technical)
print(df_technical.info())
df_technical.describe()

print(df_fundamental)
print(df_fundamental.info())
df_fundamental.describe()

print(df_securities)
print(df_securities.info())
df_securities.describe()


df_securities["GICS Sector"].value_counts()

# Filter IT sector stocks

all_tech_symbol = list(df_securities[df_securities["GICS Sector"]=="Information Technology"]["Ticker symbol"])

df_technical_tech = df_technical[df_technical["symbol"].isin(all_tech_symbol)]
symbol_close = df_technical_tech.groupby("symbol")["volume"].mean().reset_index()
symbol_close = symbol_close.sort_values(by="volume", ascending=False).reset_index(drop=True)[:3]
top_tech_symbol = list(symbol_close["symbol"])

print(df_securities[df_securities["Ticker symbol"].isin(top_tech_symbol)].reset_index(drop=True))

df_technical = df_technical[df_technical["symbol"].isin(top_tech_symbol)].reset_index(drop=True)
df_fundamental = df_fundamental[df_fundamental["Ticker Symbol"].isin(top_tech_symbol)].reset_index(drop=True)

print(df_technical)
print(df_fundamental)

# Datetime Features
print("Before Adding Datetime Features:", df_technical.shape)

col = "date"
prefix = col + "_"
df_technical[col] = pd.to_datetime(df_technical[col])
df_technical[prefix + 'year'] = df_technical[col].dt.year
df_technical[prefix + 'month'] = df_technical[col].dt.month
df_technical[prefix + 'day'] = df_technical[col].dt.day
df_technical[prefix + 'weekofyear'] = df_technical[col].dt.weekofyear
df_technical[prefix + 'dayofweek'] = df_technical[col].dt.dayofweek
df_technical[prefix + 'quarter'] = df_technical[col].dt.quarter
df_technical[prefix + 'is_month_start'] = df_technical[col].dt.is_month_start.astype(int)
df_technical[prefix + 'is_month_end'] = df_technical[col].dt.is_month_end.astype(int)

print(" After Adding Datetime Features:", df_technical.shape)

df_technical

# Lag Features
print("Before Adding Lag Features:", df_technical.shape)

ohlcv = ["open", "high", "low", "close", "volume"]
for col in ohlcv:
    lags = np.arange(65, 101, 1)
    for lag in lags:
        
        df_technical["{}_lag_{}".format(col, lag)] = df_technical.groupby("symbol")[col].transform(lambda x: x.shift(lag))

print(" After Adding Lag Features:", df_technical.shape)

df_technical

# Drop Features
print("Before Drop Features:", df_technical.shape)

drop_features = ["open", "high", "low", "volume"]
df_technical = df_technical.drop(drop_features, axis=1)

print(" After Drop Features:", df_technical.shape)

df_technical

# Datetime Features Fundamental features
print("Before Adding Datetime Features:", df_fundamental.shape)

df_fundamental["Period Ending"] = pd.to_datetime(df_fundamental["Period Ending"])
df_fundamental["First_Date_Tech"] = df_fundamental["Period Ending"] + pd.Timedelta(days=1)
df_fundamental['Last_Date_Tech'] = df_fundamental.groupby("Ticker Symbol")["Period Ending"].shift(-1)
df_fundamental['Last_Date_Tech'] = df_fundamental['Last_Date_Tech'].fillna(df_fundamental['Period Ending'] + pd.offsets.DateOffset(years=1))

print(" After Adding Datetime Features:", df_fundamental.shape)

df_fundamental

# Drop Features
print("Before Drop Features:", df_fundamental.shape)

# drop_features = ["Period Ending", "Cash Ratio", "Current Ratio", "Quick Ratio", "For Year", "Earnings Per Share", "Estimated Shares Outstanding"]
drop_features = ["Period Ending", "For Year"]
df_fundamental = df_fundamental.drop(drop_features, axis=1)

print(" After Drop Features:", df_fundamental.shape)

df_fundamental

# Filtering Date and Symbol in Technical Data
df_technical = df_technical[(df_technical["date"]>="2014-01-01")&(df_technical["date"]<="2016-12-31")].reset_index(drop=True)
# df_technical = df_technical[(df_technical["date"]>="2010-01-01")&(df_technical["date"]<="2016-12-31")].reset_index(drop=True)
df_technical

# Merging Technical and Fundamental Data
df_tech_fund = df_technical.merge(df_fundamental, left_on="symbol", right_on="Ticker Symbol")
df_tech_fund = df_tech_fund[(df_tech_fund["date"]>=df_tech_fund["First_Date_Tech"]) & (df_tech_fund["date"]<=df_tech_fund["Last_Date_Tech"])].reset_index(drop=True)

# df_tech_fund = df_technical.copy()
df_tech_fund


# Drop Features
print("Before Drop Features:", df_tech_fund.shape)

drop_features = ["Ticker Symbol", "First_Date_Tech", "Last_Date_Tech"]
df_tech_fund = df_tech_fund.drop(drop_features, axis=1)

print(" After Drop Features:", df_tech_fund.shape)

df_tech_fund

# Data Splitting (Train: 2014-Q3 2016, Test: Q4 2016)
df_train = df_tech_fund[df_tech_fund["date"] < "2016-10-01"].reset_index(drop=True)
df_test = df_tech_fund[df_tech_fund["date"] >= "2016-10-01"].reset_index(drop=True)

# Create Actual Predicted DataFrame
df_actual_predicted = pd.DataFrame({
    "date": df_train["date"],
    "symbol": df_train["symbol"],
    "actual": df_train["close"], 
})

df_actual_predicted_test = pd.DataFrame({
    "date": df_test["date"],
    "symbol": df_test["symbol"],
    "actual": df_test["close"], 
})

df_actual_predicted = df_actual_predicted.append(df_actual_predicted_test, ignore_index=True)


# Categorical Encoding
categorical_features = ["symbol"]

df_category = pd.get_dummies(data=df_train[categorical_features], drop_first=True)
df_train = pd.concat([df_train, df_category], axis=1)
df_train = df_train.drop(categorical_features, axis=1)

df_category = pd.get_dummies(data=df_test[categorical_features], drop_first=True)
df_test = pd.concat([df_test, df_category], axis=1)
df_test = df_test.drop(categorical_features, axis=1)

# Drop Date Features
df_train = df_train.drop("date", axis=1)
df_test = df_test.drop("date", axis=1)


print("Train Dataset")
print(df_train)

print("Test Dataset")
print(df_test)

X_train = df_train.drop("close", axis=1)
y_train = df_train["close"]
X_test = df_test.drop("close", axis=1)
y_test = df_test["close"]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print(" X_test:", X_test.shape)
print(" y_test:", y_test.shape)

xgb_tuning = False


def objective_xgb(trial): 
    # Parameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 25),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'random_state': 2023
    }
    
    # Fit the Model
    optuna_model = XGBRegressor(**params)
    optuna_model.fit(X_train, y_train)
    
    # Make Predictions
    y_pred = optuna_model.predict(X_test)
    
    # Evaluate Predictions
    mse = mean_squared_error(y_test, y_pred)
    return mse

if(xgb_tuning):
    xgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=2023))
    xgb_study.optimize(objective_xgb, n_trials=50)


if(xgb_tuning):
    print('Number of finished trials: {}'.format(len(xgb_study.trials)))
    print('Best trial:')
    best_xgb = xgb_study.best_trial

    print('  Value: {}'.format(best_xgb.value))
    print('  Params: ')

    for key, value in best_xgb.params.items():
        print('    {}: {}'.format(key, value))
        
    best_params = best_xgb.params
else:
    best_params = {
        "max_depth": 20,
        "learning_rate": 0.2067525580637779,
        "n_estimators": 989,
        "min_child_weight": 84,
        "gamma": 0.311037429261164,
        "subsample": 0.6539237831027682,
        "colsample_bytree": 0.45604560131724914,
        "reg_alpha": 0.6759123826338456,
        "reg_lambda": 0.4772478981932024,
    }
    print(best_params)
    
best_model = XGBRegressor(random_state=2023)
best_model.set_params(**best_params)

best_model.fit(X_train, y_train)

y_pred_train = best_model.predict(X_train)
n = X_train.shape[0]
p = X_train.shape[1]


def adjusted_r2_score(y_test, y_pred, n, p):
    score = r2_score(y_test, y_pred)
    return 1 - (1 - score) * (n - 1) / (n - p - 1)

mse = mean_squared_error(y_train, y_pred_train)
rmse = mean_squared_error(y_train, y_pred_train, squared=False)
mae = mean_absolute_error(y_train, y_pred_train)
r2 = r2_score(y_train, y_pred_train)
adj_r2 = adjusted_r2_score(y_train, y_pred_train, n, p)

print("Metric Scores for Training")
print("MSE    :", mse)
print("RMSE   :", rmse)
print("MAE    :", mae)
print("R2     :", r2)
print("Adj R2 :", adj_r2)

y_pred = best_model.predict(X_test)
n = X_test.shape[0]
p = X_test.shape[1]


mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = adjusted_r2_score(y_test, y_pred, n, p)


print("Metric Scores for Testing")
print("MSE    :", mse)
print("RMSE   :", rmse)
print("MAE    :", mae)
print("R2     :", r2)
print("Adj R2 :", adj_r2)


all_y_pred = np.concatenate((y_pred_train, y_pred))
df_actual_predicted["predicted"] = all_y_pred
df_actual_predicted_train = df_actual_predicted[df_actual_predicted["date"]<"2016-10-01"].reset_index(drop=True)
df_actual_predicted_test = df_actual_predicted[df_actual_predicted["date"]>="2016-10-01"].reset_index(drop=True)


print(df_actual_predicted)
print(df_actual_predicted_train)
print(df_actual_predicted_test)

# Create Subplots
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=["Apple: All Closing Prices", "Apple: Test Closing Prices", 
                    "Intel: All Closing Prices", "Intel: Test Closing Prices",
                    "Microsoft: All Closing Prices", "Microsoft: Test Closing Prices",
                   ], 
)

# Line Plot
for i in range(3):
    df_symbol_all = df_actual_predicted[df_actual_predicted["symbol"]==top_tech_symbol[i]]
    df_symbol_train = df_actual_predicted_train[df_actual_predicted_train["symbol"]==top_tech_symbol[i]]
    df_symbol_test = df_actual_predicted_test[df_actual_predicted_test["symbol"]==top_tech_symbol[i]]
    
    mse_train = round(mean_squared_error(df_symbol_train["actual"], df_symbol_train["predicted"]), 2)
    mse_test = round(mean_squared_error(df_symbol_test["actual"], df_symbol_test["predicted"]), 2)
    
    min_price = df_symbol_all["actual"].min()
    max_price = df_symbol_all["actual"].max()
    
    # All Data
    # Actual
    fig.add_trace(
        go.Scatter(
            x=df_symbol_all["date"],
            y=df_symbol_all["actual"],
            mode="lines",
            line=dict(
                width=1,
                color="#073b4c"
            ),
            name="Actual"
        ), row=i+1, col=1
    )
    
    # Train Predicted
    fig.add_trace(
        go.Scatter(
            x=df_symbol_train["date"],
            y=df_symbol_train["predicted"],
            mode="lines",
            line=dict(
                width=1.5,
                color="#ef476f"
            ),
            name="Train Predicted"
        ), row=i+1, col=1
    )
    
    # Test Predicted
        fig.add_trace(
            go.Scatter(
                x=df_symbol_test["date"],
                y=df_symbol_test["predicted"],
                mode="lines",
                line=dict(
                    width=1.5,
                    color="#06d6a0"
                ),
                name="Test Predicted"
            ), row=i+1, col=1
        )
        
        # Annotation
        fig.add_annotation(
            x="2016-03-01", 
            y=min_price + 0.1 * (max_price - min_price), 
            xref="x{}".format(i*2+1), yref="y{}".format(i*2+1), xanchor="left",
            text="<b>MSE Train: {}</b><br><b>MSE Test : {}</b>".format(mse_train, mse_test),
            font=dict(
                color="#073b4c",
                size=12,
            ),
            showarrow=False
        )
        
        # Test Data
        # Actual
        fig.add_trace(
            go.Scatter(
                x=df_symbol_test["date"],
                y=df_symbol_test["actual"],
                mode="lines",
                line=dict(
                    width=2,
                    color="#073b4c"
                ),
                name="Actual"
            ), row=i+1, col=2
        )
        
        # Predicted
        fig.add_trace(
            go.Scatter(
                x=df_symbol_test["date"],
                y=df_symbol_test["predicted"],
                mode="lines",
                line=dict(
                    width=1.5,
                    color="#06d6a0"
                ),
                name="Predicted"
            ), row=i+1, col=2
        )
        
        # Update Axes
        fig.update_xaxes(linecolor="Black", ticks="outside", row=i+1, col=1)
        fig.update_xaxes(linecolor="Black", ticks="outside", row=i+1, col=2)
        fig.update_yaxes(linecolor="Black", ticks="outside", row=i+1, col=1)
        fig.update_yaxes(linecolor="Black", ticks="outside", row=i+1, col=2)
    
# Update Layout
fig.update_layout(
    title="<b>Closing Prices per Stock</b>", title_x=0.5, font_family="Garamond", font_size=14,
    width=950, height=850,
    showlegend=False,
    plot_bgcolor="White",
    paper_bgcolor="White"
)

# Show
fig.show(renderer="iframe_connected")



#  Residual Plot 


df_actual_predicted_train["residual"] = df_actual_predicted_train["actual"] - df_actual_predicted_train["predicted"]
df_actual_predicted_test["residual"] = df_actual_predicted_test["actual"] - df_actual_predicted_test["predicted"]


# Create Subplots
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=["Apple: Train Residual", "Apple: Test Residual", 
                    "Intel: Train Residual", "Intel: Test Residual",
                    "Microsoft: Train Residual", "Microsoft: Test Residual",
                   ], 
)

# Line Plot
for i in range(3):
    df_symbol_train = df_actual_predicted_train[df_actual_predicted_train["symbol"]==top_tech_symbol[i]]
    df_symbol_test = df_actual_predicted_test[df_actual_predicted_test["symbol"]==top_tech_symbol[i]]
    
    # All Data
    # Train
    fig.add_trace(
        go.Scatter(
            x=df_symbol_train["actual"],
            y=df_symbol_train["residual"],
            mode="markers",
            marker=dict(
                color="#ef476f",
                size=6
            ),
            name="Residual"
        ), row=i+1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[df_symbol_train["actual"].min(), df_symbol_train["actual"].max()],
            y=[0.0, 0.0],
            mode="lines",
            line=dict(
                color="#073b4c",
                width=3
            ),
            name="Actual"
        ), row=i+1, col=1
    )
    
        
    # Test
    fig.add_trace(
        go.Scatter(
            x=df_symbol_test["actual"],
            y=df_symbol_test["residual"],
            mode="markers",
            marker=dict(
                color="#06d6a0",
                size=6
            ),
            name="Residual"
        ), row=i+1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[df_symbol_test["actual"].min(), df_symbol_test["actual"].max()],
            y=[0.0, 0.0],
            mode="lines",
            line=dict(
                color="#073b4c",
                width=3
            ),
            name="Actual"
        ), row=i+1, col=2
    )
    
    # Update Axes
    fig.update_xaxes(linecolor="Black", ticks="outside", row=i+1, col=1)
    fig.update_xaxes(linecolor="Black", ticks="outside", row=i+1, col=2)
    fig.update_yaxes(linecolor="Black", ticks="outside", row=i+1, col=1)
    fig.update_yaxes(linecolor="Black", ticks="outside", row=i+1, col=2)

# Update Layout
fig.update_layout(
    title="<b>Residual Plot</b>", title_x=0.5, font_family="Garamond", font_size=14,
    width=950, height=850,
    showlegend=False,
    plot_bgcolor="White",
    paper_bgcolor="White"
)    

# Show
fig.show(renderer="iframe_connected")

# Feature Importance 

# Create DataFrame
df_importance = pd.DataFrame()
df_importance["Features"] = X_train.columns
df_importance["Importance"] = best_model.feature_importances_
df_importance = df_importance.sort_values(by="Importance", ascending=False).reset_index(drop=True)[:20]
df_importance = df_importance.sort_values(by="Importance", ascending=True).reset_index(drop=True)


# Create Figure
fig = go.Figure()

# Bar Plot
fig.add_trace(
    go.Bar(
        x=df_importance["Importance"],
        y=df_importance["Features"],
        orientation="h",
        name="Importance",
        marker_color="#118ab2",
        width=0.8
    )
)

# Annotations
for i in range(20):
    fig.add_annotation(
        x=df_importance["Importance"][i] + 0.003, y=df_importance["Features"][i], 
        xref="x1", yref="y1", xanchor="left",
        text="<b>{}</b>".format(df_importance["Features"][i]),
        font=dict(
            color="#118ab2",
            size=12,
        ),
        showarrow=False
    )

# Update Axes
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

# Update Layout
fig.update_layout(
    title="<b>Top 20 Feature Importance</b>", title_x=0.5,
    font_family="Garamond", font_size=14,
    width=950, height=900,
    plot_bgcolor="White",
    showlegend=False
)

# Show
fig.show(renderer="iframe_connected")
