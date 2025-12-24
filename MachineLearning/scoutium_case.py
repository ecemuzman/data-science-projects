############################################################
# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ
############################################################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


############################################################
# 2. VERİ SETLERİNİN OKUTULMASI
############################################################
# Oyuncuların maç içi özellik puanları
scoutium_attributes = pd.read_csv(
    "machine_learning/datasets/scoutium_attributes.csv", sep=";"
)

# Oyuncuların scout tarafından verilen potansiyel etiketleri
scoutium_labels = pd.read_csv(
    "machine_learning/datasets/scoutium_potential_labels.csv", sep=";"
)


############################################################
# 3. VERİ SETLERİNİN BİRLEŞTİRİLMESİ
############################################################
# Ortak id’ler üzerinden iki veri seti birleştirildi
df = scoutium_attributes.merge(
    scoutium_labels,
    on=["task_response_id", "match_id", "evaluator_id", "player_id"]
)


############################################################
# 4. KALECİ POZİSYONUNUN ÇIKARILMASI
############################################################
# Kaleciler (position_id = 1) analiz dışında bırakıldı
df = df[df["position_id"] != 1]


############################################################
# 5. BELOW_AVERAGE ETİKETİNİN ÇIKARILMASI
############################################################
# Veri setinde çok az sayıda bulunan sınıf çıkarıldı
df = df[df["potential_label"] != "below_average"]


############################################################
# 6. PIVOT TABLE OLUŞTURULMASI
############################################################
# Her satır bir oyuncu olacak şekilde veri seti yeniden şekillendirildi
new_df = df.pivot_table(
    index=["player_id", "position_id", "potential_label"],
    columns="attribute_id",
    values="attribute_value"
)

# Index değişken haline getirildi
new_df.reset_index(inplace=True)

# Attribute id’ler string formata çevrildi
new_df.columns = new_df.columns.astype(str)


############################################################
# 7. HEDEF DEĞİŞKENİN SAYISALLAŞTIRILMASI
############################################################
# average -> 0, highlighted -> 1
le = LabelEncoder()
new_df["potential_label"] = le.fit_transform(new_df["potential_label"])


############################################################
# 8. SAYISAL DEĞİŞKENLERİN BELİRLENMESİ
############################################################
# Modelde kullanılacak sayısal değişkenler seçildi
num_cols = new_df.columns.difference(
    ["player_id", "position_id", "potential_label"]
)


############################################################
# 9. STANDARD SCALER UYGULANMASI
############################################################
# Sayısal değişkenler ölçeklendirildi
scaler = StandardScaler()
new_df[num_cols] = scaler.fit_transform(new_df[num_cols])


############################################################
# 10. BAĞIMSIZ VE BAĞIMLI DEĞİŞKENLER
############################################################
X = new_df.drop("potential_label", axis=1)
y = new_df["potential_label"]


############################################################
# 11. BASE MODEL KARŞILAŞTIRMASI
############################################################
def base_models_tabular(X, y):

    scoring = ["roc_auc", "accuracy", "precision", "recall", "f1"]

    classifiers = [
        ('LR', LogisticRegression(max_iter=1000)),
        ('KNN', KNeighborsClassifier()),
        ('SVC', SVC(probability=True)),
        ('CART', DecisionTreeClassifier()),
        ('RF', RandomForestClassifier()),
        ('AdaBoost', AdaBoostClassifier()),
        ('GBM', GradientBoostingClassifier()),
        ('XGBoost', XGBClassifier(eval_metric='logloss')),
        ('LightGBM', LGBMClassifier(verbosity=-1))
    ]

    results = []

    for name, model in classifiers:
        cv_results = cross_validate(model, X, y, cv=3, scoring=scoring)
        results.append({
            "Model": name,
            "ROC_AUC": cv_results["test_roc_auc"].mean(),
            "ACCURACY": cv_results["test_accuracy"].mean(),
            "PRECISION": cv_results["test_precision"].mean(),
            "RECALL": cv_results["test_recall"].mean(),
            "F1": cv_results["test_f1"].mean()
        })

    return pd.DataFrame(results)


base_models_tabular(X, y)


############################################################
# 12. RANDOM FOREST MODELİ VE TEST SONUCU
############################################################
# Train - Test ayrımı yapıldı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=17
)

# Random Forest modeli kuruldu ve eğitildi
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapıldı
y_pred = rf_model.predict(X_test)

# Model doğruluk skoru
accuracy_score(y_test, y_pred)


############################################################
# 13. FEATURE IMPORTANCE GÖRSELLEŞTİRME
############################################################
def plot_importance(model, X):

    feature_imp = pd.DataFrame({
        "Value": model.feature_importances_,
        "Feature": X.columns
    }).sort_values(by="Value", ascending=False)

    plt.figure(figsize=(10, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()


plot_importance(rf_model, X_train)
