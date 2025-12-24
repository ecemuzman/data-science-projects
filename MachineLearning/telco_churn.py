############################################################
# 1. GEREKLİ KÜTÜPHANELER
############################################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


############################################################
# 2. VERİ SETİNİN OKUTULMASI
############################################################
df = pd.read_csv("machine_learning/datasets/Telco-Customer-Churn.csv")


############################################################
# 3. KEŞİFÇİ VERİ ANALİZİ (EDA)
############################################################
def check_df(dataframe, head=5):
    print("Shape:", dataframe.shape)
    print("\nTypes:\n", dataframe.dtypes)
    print("\nHead:\n", dataframe.head(head))
    print("\nMissing Values:\n", dataframe.isnull().sum())
    print("\nQuantiles:\n", dataframe.describe([0.05, 0.50, 0.95, 0.99]).T)

check_df(df)


############################################################
# 4. VERİ TİPİ DÜZENLEMELERİ
############################################################
# Hedef değişken sayısallaştırıldı
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("object")

# SeniorCitizen kategorik olarak ele alındı
df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")

# TotalCharges sayısal tipe çevrildi, hatalı değerler NaN yapıldı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Eksik TotalCharges değerleri 0 ile dolduruldu
df["TotalCharges"] = df["TotalCharges"].fillna(0)


############################################################
# 5. KATEGORİK VE SAYISAL DEĞİŞKENLER
############################################################
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
num_cols = [col for col in df.columns if df[col].dtype in ["int", "float"]]


############################################################
# 6. HEDEF DEĞİŞKEN & KATEGORİK DEĞİŞKEN ANALİZİ
############################################################
def target_summary_with_cat(dataframe, target, col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(col)[target].mean()}))

for col in cat_cols:
    if col != "Churn":
        target_summary_with_cat(df, "Churn", col)


############################################################
# 7. KORELASYON ANALİZİ
############################################################
corr = df[num_cols].corr()
sns.heatmap(corr, cmap="RdBu")
plt.show()


############################################################
# 8. FEATURE ENGINEERING
############################################################
# CustomerID sayısallaştırıldı
df["customerID"] = df["customerID"].factorize()[0]

# Müşteri kalma süresi kategorize edildi
df["TenureCat"] = pd.cut(df["tenure"],
                         bins=[-1, 12, 24, 48, 72],
                         labels=["0-1 year", "1-2 year", "2-4 year", "4-6 year"])

# Ödeme tipleri sadeleştirildi
df["PaymentType"] = df["PaymentMethod"].replace({
    "Electronic check": "electronic",
    "Bank transfer (automatic)": "automatic",
    "Credit card (automatic)": "automatic",
    "Mailed check": "manual"
})

# Alınan servis sayısı hesaplandı
services = ["OnlineBackup", "OnlineSecurity", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]

df["NumServices"] = df[services].apply(lambda x: (x == "Yes").sum(), axis=1)

# Log dönüşümü
df["TotalCharges_log"] = np.log1p(df["TotalCharges"])

# İnternet hizmeti var mı?
df["Has_Internet"] = (df["InternetService"] != "No").astype(int)


############################################################
# 9. ENCODING İŞLEMLERİ
############################################################
# Binary değişkenler için Label Encoding
binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == "O"]

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# No internet service ifadeleri düzenlendi
replace_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

for col in replace_cols:
    df[col] = df[col].replace("No internet service", "No")

# One-Hot Encoding
ohe_cols = [col for col in df.columns if 2 < df[col].nunique() <= 10]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)


############################################################
# 10. ÖLÇEKLENDİRME
############################################################
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


############################################################
# 11. LOGISTIC REGRESSION MODELİ
############################################################
X = df.drop(["Churn"], axis=1)
y = df["Churn"]

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X, y)

y_pred = log_model.predict(X)

print(classification_report(y, y_pred))


############################################################
# 12. CATBOOST MODELİ
############################################################
from catboost import CatBoostClassifier

dff = pd.read_csv("machine_learning/datasets/Telco-Customer-Churn.csv")
dff["Churn"] = dff["Churn"].map({"Yes": 1, "No": 0})

cat_features = [col for col in dff.columns if dff[col].dtype == "object" and col != "Churn"]

X = dff.drop("Churn", axis=1)
y = dff["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

y_prob = cat_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.35).astype(int)

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
