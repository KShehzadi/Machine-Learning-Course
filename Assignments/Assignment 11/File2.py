# -------------------------- ---------- Split Train Test  --------------------
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import cufflinks as cf
cf.set_config_file(offline=True, sharing=False, theme='ggplot')
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# -------------------------- The Data ---------------------------
from seaborn import load_dataset
data = load_dataset("mpg")
tr, te = train_test_split(data, test_size=0.25, random_state=83)

# -------------------------- Building a Few Basic Models --------
models = {}
quantitative_features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
for i in range(len(quantitative_features)):
    # -------------------------- The features to include in the ith model
    features = quantitative_features[:(i+1)]
    # -------------------------- The name we are giving to the ith model
    name = ",".join([name[0] for name in features])
    # -------------------------- The pipeline for the ith model
    model = Pipeline([
        ("SelectColumns", ColumnTransformer([
            ("keep", "passthrough", features),
        ])),
        ("Imputation", SimpleImputer()),
        ("LinearModel", LinearRegression())
    ])
    # -------------------------- Fit the pipeline
    model.fit(tr, tr['mpg'])
    # -------------------------- Saving the ith model
    models[name] = model
print(models.keys())

# -------------------------- cross_val_score --------------------
from sklearn.model_selection import cross_val_score
def rmse_score(model, X, y):
    return np.sqrt(np.mean((y - model.predict(X))**2))
cross_val_score(models['c'], tr, tr['mpg'], scoring=rmse_score, cv=5)
print(np.mean(cross_val_score(models['c'], tr, tr['mpg'], scoring=rmse_score, cv=5)))

# -------------------------- Visualizing the Train/CV/Test RMSE -
def compare_models(models):
    # -------------------------- Compute the training error for each model
    training_rmse = [rmse_score(model, tr, tr['mpg']) for model in models.values()]
    # -------------------------- Compute the cross validation error for each model
    validation_rmse = [np.mean(cross_val_score(model, tr, tr['mpg'], scoring=rmse_score, cv=5)) 
                       for model in models.values()]
    # -------------------------- Compute the test error for each model (don't do this!)
    test_rmse = [rmse_score(model, te, te['mpg']) for model in models.values()]
    names = list(models.keys())
    fig = go.Figure([
        go.Bar(x = names, y = training_rmse, name="Training RMSE"),
        go.Bar(x = names, y = validation_rmse, name="CV RMSE"),
        go.Bar(x = names, y = test_rmse, name="Test RMSE", opacity=.3)])
    fig.update_yaxes(title="RMSE")
    return fig

compare_models(models)

# -------------------------- Adding the Text Features ------------
# -------------------------- Adding the Origin ------------------
print(tr['origin'].value_counts())
from sklearn.preprocessing import OneHotEncoder
model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", "passthrough", quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"])
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", LinearRegression())
])
model.fit(tr, tr['mpg'])
name = ",".join([name[0] for name in quantitative_features]) + ",o"
models[name] = model

# -------------------------- Adding the Vehicle Name ------------
from sklearn.feature_extraction.text import CountVectorizer
model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", "passthrough", quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", LinearRegression())
])
model.fit(tr, tr['mpg'])
name = ",".join([name[0] for name in quantitative_features]) + ",o,n"
models[name] = model
compare_models(models)
fig = go.Figure()
fig.add_trace(go.Bar(
    x = list(models.keys()), 
    y = [len(models[m]["LinearModel"].coef_) for m in models]
))
fig.update_yaxes(title="Number of Features",type="log")

# -------------------------- Ridge Regression in SK Learn --------
from sklearn.linear_model import Ridge
ridge_model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", "passthrough", quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", Ridge(alpha=0.5))
])
ridge_model.fit(tr, tr['mpg'])
models["Ridge(alpha=0.5)"] = ridge_model
compare_models(models)
from sklearn.preprocessing import StandardScaler
ridge_model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", StandardScaler(), quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
# --------------------------     ("Standarize", StandardScaler(with_mean=False)),
    ("LinearModel", Ridge(alpha=0.5))
])
ridge_model.fit(tr, tr['mpg'])
models["RidgeN(alpha=0.5)"] = ridge_model
compare_models(models)
ridge_model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", StandardScaler(), quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", Ridge(alpha=10))
])
ridge_model.fit(tr, tr['mpg'])
models["RidgeN(alpha=10)"] = ridge_model
compare_models(models)
ridge_model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", StandardScaler(), quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", Ridge(alpha=10))
])

alphas = np.linspace(0.5, 20, 30)
cv_values = []
train_values = []
test_values = []
for alpha in alphas:
    ridge_model.set_params(LinearModel__alpha=alpha)
    cv_values.append(np.mean(cross_val_score(ridge_model, tr, tr['mpg'], scoring=rmse_score, cv=5)))
    ridge_model.fit(tr, tr['mpg'])
    train_values.append(rmse_score(ridge_model, tr, tr['mpg']))
    test_values.append(rmse_score(ridge_model, te, te['mpg']))
fig = go.Figure()
fig.add_trace(go.Scatter(x = alphas, y = train_values, mode="lines+markers", name="Train"))
fig.add_trace(go.Scatter(x = alphas, y = cv_values, mode="lines+markers", name="CV"))
fig.add_trace(go.Scatter(x = alphas, y = test_values, mode="lines+markers", name="Test"))
fig.update_layout(xaxis_title=r"$\alpha$", yaxis_title="CV RMSE")
best_alpha = alphas[np.argmin(cv_values)]
ridge_model.set_params(LinearModel__alpha=best_alpha)
ridge_model.fit(tr, tr['mpg'])
models["RidgeN(alpha_best)"] = ridge_model
compare_models(models)
from sklearn.linear_model import RidgeCV
alphas = np.linspace(0.5, 3, 30)

ridge_model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", StandardScaler(), quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", RidgeCV(alphas=alphas))
])
ridge_model.fit(tr, tr['mpg'])
models["RidgeCV"] = ridge_model
compare_models(models)

# -------------------------- Lasso in SKLearn -------------------
from sklearn.linear_model import Lasso, LassoCV
lasso_model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", StandardScaler(), quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", LassoCV(cv=3))
])
lasso_model.fit(tr, tr['mpg'])
models["LassoCV"] = lasso_model
compare_models(models)
ff.create_distplot([
    models['LassoCV']["LinearModel"].coef_, 
    models['RidgeCV']["LinearModel"].coef_],
    ["Lasso", "Ridge"], bin_size=0.1)
ct = models['LassoCV']['SelectColumns']
feature_names = (
    quantitative_features +
    list(ct.named_transformers_['origin_encoder'].get_feature_names())+
    list(ct.named_transformers_['text'].get_feature_names())
)
feature_names = np.array(feature_names)
print(feature_names)
kept = ~np.isclose(models['LassoCV']["LinearModel"].coef_, 0)
feature_names[kept]