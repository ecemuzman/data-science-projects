#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

# Gerekli kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#############################################
# VERİ SETLERİNİN OKUTULMASI VE BİRLEŞTİRİLMESİ
#############################################

# Train ve test veri setlerini okuma
train = pd.read_csv("machine_learning/datasets/house_pricing/train (1).csv")
test = pd.read_csv("machine_learning/datasets/house_pricing/test (1).csv")

# İki veri setini birleştirerek tek dataframe oluşturma
df = pd.concat([train, test])

#############################################
# DEĞİŞKEN TÜRLERİNİN BELİRLENMESİ
#############################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kardinal değişkenleri ayırır
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns
                   if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns
                if dataframe[col].dtype != "O" and col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#############################################
# ID DEĞİŞKENİNİN SİLİNMESİ
#############################################

df.drop(columns=["Id"], inplace=True)

#############################################
# TİP DÖNÜŞÜMLERİ
#############################################

# Sayısal görünümlü kategorik değişkenleri stringe çevirme
num_but_cat = ["MSSubClass", "MoSold", "YrSold"]
df[num_but_cat] = df[num_but_cat].astype(str)

#############################################
# EKSİK DEĞER ANALİZİ VE DOLDURMA
#############################################

# Sayısal değişkenlerde medyan ile doldurma
for col in num_cols:
    if col != "SalePrice":
        df[col] = df[col].fillna(df[col].median())

# Kategorik değişkenlerde mod ile doldurma
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

#############################################
# ORDINAL DEĞİŞKENLERİN SAYISALLAŞTIRILMASI
#############################################

qual_map = {"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}

ordinal_cols = [
    "ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
    "KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"
]

for col in ordinal_cols:
    df[col] = df[col].map(qual_map).astype(int)

#############################################
# ONE-HOT ENCODING
#############################################

ohe_cols = [col for col in df.columns
            if df[col].dtype == "object" and 2 < df[col].nunique() <= 10]

df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# Kalan object kolonları da encode et
df = pd.get_dummies(df, drop_first=True)

#############################################
# TRAIN / TEST AYRIMI
#############################################

train_df = df[df["SalePrice"].notnull()]
test_df = df[df["SalePrice"].isnull()]

X = train_df.drop("SalePrice", axis=1)
y = train_df["SalePrice"]
X_test = test_df.drop("SalePrice", axis=1)

#############################################
# XGBOOST MODELİ
#############################################

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cv_results = cross_validate(
    xgb, X, y,
    cv=5,
    scoring=("r2", "neg_root_mean_squared_error")
)

print("XGBoost R²:", cv_results["test_r2"].mean())
print("XGBoost RMSE:", -cv_results["test_neg_root_mean_squared_error"].mean())

#############################################
# LOG DÖNÜŞÜMÜ VE CATBOOST MODELİ
#############################################

y_log = np.log1p(y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_log, test_size=0.20, random_state=42
)

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=500,
    od_type="Iter",
    od_wait=100
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

# Validation performansı
y_valid_pred = np.expm1(cat_model.predict(X_valid))
y_valid_real = np.expm1(y_valid)

print("CatBoost R²:", r2_score(y_valid_real, y_valid_pred))
print("CatBoost RMSE:", mean_squared_error(y_valid_real, y_valid_pred))

#############################################
# FINAL MODEL VE TEST TAHMİNLERİ
#############################################

cat_model_final = CatBoostRegressor(
    loss_function="RMSE",
    iterations=cat_model.tree_count_,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=False
)

cat_model_final.fit(X, y_log)

test_pred_log = cat_model_final.predict(X_test)
test_pred = np.expm1(test_pred_log)
