# ------------------------- Split Train and Test Data  --------------------
# ------------------------- Using Pandas Functions------------
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import cufflinks as cf
cf.set_config_file(offline=True, sharing=False, theme='ggplot')
from sklearn.linear_model import LinearRegression
from seaborn import load_dataset
data = load_dataset("mpg")
shuffled_data = data.sample(frac=1., random_state=42)
split_point = int(shuffled_data.shape[0]*0.90)
tr = shuffled_data.iloc[:split_point]
te = shuffled_data.iloc[split_point:]

# ------------------------- Using SKLearn ----------------------
from sklearn.model_selection import train_test_split
tr, te = train_test_split(data, test_size=0.1, random_state=83)
ff.create_distplot([tr['mpg'], te['mpg']], ['train mpg', 'test mpg'])

# ------------------------- Building A Basic Model --------------
def phi(df):
    return df[["cylinders", "displacement"]]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(phi(tr), tr['mpg'])
def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))
Y_hat = model.predict(phi(tr))
Y = tr['mpg']
print("Training Error (RMSE):", rmse(Y, Y_hat))

# ------------------------- SKLearn Pipelines -------------------
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
model = Pipeline([
    ("SelectColumns", ColumnTransformer([("keep", "passthrough", ["cylinders", "displacement"])])),
    ("LinearModel", LinearRegression())
])
model.fit(tr, tr['mpg'])
models = {"c+d": model}

# ------------------------- More Feature Transformations --------
from sklearn.preprocessing import FunctionTransformer
def compute_volume(X):
    return np.expand_dims(X[:,1] / X[:,0]  , axis=1)
volume_transformer = FunctionTransformer(compute_volume, validate=True)
model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", "passthrough", ["cylinders", "displacement"]),
        ("cyl_vol", volume_transformer, ["cylinders", "displacement"])])),
    ("LinearModel", LinearRegression())
])
model.fit(tr, tr['mpg'])
Y_hat = model.predict(tr)
Y = tr['mpg']
print("Training Error (RMSE):", rmse(Y, Y_hat))
models["c+d+d/c"] = model

# ------------------------- Adding More Features ----------------
quantitative_features = ["cylinders", "displacement", "horsepower", "weight", "acceleration"]
model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", "passthrough", quantitative_features),
        ("cyl_vol", volume_transformer, ["cylinders", "displacement"])])),
    ("LinearModel", LinearRegression())
])
try:
    model.fit(tr, tr['mpg'])
except ValueError as err:
    print(err)

from sklearn.impute import SimpleImputer
model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", "passthrough", quantitative_features),
        ("cyl_vol", volume_transformer, ["cylinders", "displacement"])])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", LinearRegression())
])
model.fit(tr, tr['mpg'])
models['c+d+d/c+h+w+a'] = model
Y_hat = model.predict(tr)
Y = tr['mpg']
print("Training Error (RMSE):", rmse(Y, Y_hat))

# ------------------------- Cross Validation ---------------------
from sklearn.model_selection import KFold
from sklearn.base import clone

def cross_validate_rmse(model):
    model = clone(model)
    five_fold = KFold(n_splits=5)
    rmse_values = []
    for tr_ind, va_ind in five_fold.split(tr):
        model.fit(tr.iloc[tr_ind,:], tr['mpg'].iloc[tr_ind])
        rmse_values.append(rmse(tr['mpg'].iloc[va_ind], model.predict(tr.iloc[va_ind,:])))
    return np.mean(rmse_values)

cross_validate_rmse(model)

def compare_models(models):
    # ------------------------- Compute the training error for each model
    training_rmse = [rmse(tr['mpg'], model.predict(tr)) for model in models.values()]
    # ------------------------- Compute the cross validation error for each model
    validation_rmse = [cross_validate_rmse(model) for model in models.values()]
    # ------------------------- Compute the test error for each model (don't do this!)
    test_rmse = [rmse(te['mpg'], model.predict(te)) for model in models.values()]
    names = list(models.keys())
    fig = go.Figure([
        go.Bar(x = names, y = training_rmse, name="Training RMSE"),
        go.Bar(x = names, y = validation_rmse, name="CV RMSE"),
        go.Bar(x = names, y = test_rmse, name="Test RMSE", opacity=.3)])
    return fig

fig = compare_models(models)
fig.update_yaxes(range=[2,5.1], title="RMSE")

quantitative_features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", "passthrough", quantitative_features),
        ("cyl_vol", volume_transformer, ["cylinders", "displacement"])
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", LinearRegression())
])

model.fit(tr, tr['mpg'])
models['c+d+d/c+h+w+a+y'] = model

fig = compare_models(models)
fig.update_yaxes(range=[2,5.1], title="RMSE")

# ------------------------- Going too Far? -----------------------
from sklearn.feature_extraction.text import CountVectorizer
model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", "passthrough", quantitative_features),
        ("cyl_vol", volume_transformer, ["cylinders", "displacement"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", LinearRegression())
])
model.fit(tr, tr['mpg'])
models['c+d+d/c+h+w+a+y+n'] = model
fig = compare_models(models)
fig.update_yaxes(range=[0,5.1], title="RMSE")
best_model = clone(models['c+d+d/c+h+w+a+y'])
best_model.fit(data, data['mpg'])
print(rmse(best_model.predict(te), te['mpg']))