############################################################
# 1. GEREKLİ KÜTÜPHANELER
############################################################
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


############################################################
# 2. VERİ SETİNİN OKUTULMASI
############################################################
df = pd.read_csv("Datasets/flo_data_20k.csv")


############################################################
# 3. VERİ HAZIRLAMA (DATA PREP)
############################################################
def data_prep(dataframe):
    # Toplam sipariş sayısı
    dataframe["order_num_total"] = (
        dataframe["order_num_total_ever_online"] +
        dataframe["order_num_total_ever_offline"]
    )

    # Toplam müşteri değeri
    dataframe["customer_value_total"] = (
        dataframe["customer_value_total_ever_online"] +
        dataframe["customer_value_total_ever_offline"]
    )

    # Tarih değişkenlerini datetime formatına çevirme
    date_cols = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_cols] = dataframe[date_cols].apply(pd.to_datetime)

    return dataframe

df = data_prep(df)


############################################################
# 4. ZAMANA DAYALI DEĞİŞKENLER
############################################################
# Analiz tarihi
today_date = df["last_order_date"].max() + dt.timedelta(days=2)

# Recency: Son alışverişten bu yana geçen gün
df["recency"] = (today_date - df["last_order_date"]).dt.days

# Tenure: Müşterinin şirkette kaldığı süre
df["tenure"] = (today_date - df["first_order_date"]).dt.days


############################################################
# 5. DAVRANIŞSAL FEATURE ENGINEERING
############################################################
# Ortalama sipariş tutarı
df["avg_order_value"] = df["customer_value_total"] / df["order_num_total"]

# Satın alma sıklığı
df["purchase_freq"] = df["order_num_total"] / df["tenure"]

# Online – offline oranları
df["online_ratio"] = df["order_num_total_ever_online"] / df["order_num_total"]
df["offline_ratio"] = df["order_num_total_ever_offline"] / df["order_num_total"]

# İlgi duyulan kategori sayısı
df["category_count"] = df["interested_in_categories_12"].str.count(",") + 1


############################################################
# 6. KÜMELEMEDE KULLANILACAK DEĞİŞKENLER
############################################################
X = df[[
    "recency",
    "tenure",
    "order_num_total",
    "customer_value_total"
]]


############################################################
# 7. VERİ ÖLÇEKLENDİRME (STANDARD SCALER)
############################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


############################################################
# 8. K-MEANS – OPTİMUM K BELİRLEME
############################################################
# Manuel Elbow Yöntemi
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_scaled_df)
    ssd.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, ssd, "bx-")
plt.xlabel("K Değeri")
plt.ylabel("SSD")
plt.title("Elbow Yöntemi ile Optimum K")
plt.show()

# Yellowbrick ile otomatik Elbow
kmeans = KMeans(random_state=42, n_init="auto")
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(X_scaled_df)
elbow.show()

optimal_k = elbow.elbow_value_


############################################################
# 9. K-MEANS MODELİ VE SEGMENTASYON
############################################################
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled_df) + 1

# Küme özetleri
kmeans_summary = df.groupby("KMeans_Cluster").agg({
    "recency": ["mean", "median"],
    "tenure": ["mean", "median"],
    "order_num_total": ["mean", "median"],
    "customer_value_total": ["mean", "median"]
})
print(kmeans_summary)


############################################################
# 10. PCA İLE K-MEANS GÖRSELLEŞTİRME
############################################################
pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
pca_df["KMeans_Cluster"] = df["KMeans_Cluster"]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="KMeans_Cluster",
    palette="tab10",
    alpha=0.7
)
plt.title("K-Means Kümeleri (PCA)")
plt.show()


############################################################
# 11. HİYERARŞİK KÜMELEME (DENDROGRAM)
############################################################
# HC için MinMax ölçekleme
scaler_hc = MinMaxScaler()
X_scaled_hc = scaler_hc.fit_transform(X)

# Büyük veri için örneklem
sample_size = min(len(X_scaled_hc), 500)
sample = pd.DataFrame(X_scaled_hc, columns=X.columns).sample(sample_size, random_state=42)

# Ward Linkage dendrogram
Z = linkage(sample, method="ward")

plt.figure(figsize=(12, 6))
plt.title("Hiyerarşik Kümeleme Dendrogramı (Ward)")
dendrogram(Z, truncate_mode="lastp", p=10)
plt.show()


############################################################
# 12. AGGLOMERATIVE CLUSTERING
############################################################
n_clusters_hc = 4
hc = AgglomerativeClustering(n_clusters=n_clusters_hc, linkage="ward")
df["HC_Cluster"] = hc.fit_predict(X_scaled_hc) + 1

hc_summary = df.groupby("HC_Cluster").agg({
    "recency": ["mean", "median"],
    "tenure": ["mean", "median"],
    "customer_value_total": ["mean", "median"]
})
print(hc_summary)

