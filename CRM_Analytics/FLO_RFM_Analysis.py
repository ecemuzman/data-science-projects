
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
# omni_channel : Toplam alışveriş sayısı hem online hem offline
# omni_channel_total : Toplam alışveriş tutarı
###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.
           # 2. Veri setinde
                     # a. İlk 10 gözlem,
                     # b. Değişken isimleri,
                     # c. Betimsel istatistik,
                     # d. Boş değer,
                     # e. Değişken tipleri, incelemesi yapınız.
           # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
           # 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
           # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
           # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

# GÖREV 2: RFM Metriklerinin Hesaplanması

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.


           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
                   # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.

             # b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
                   # alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
                   # olarak kaydediniz.

# GÖREV 6: Tüm süreci fonksiyonlaştırınız.


###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################

import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("/Users/ecemuzman/PycharmProjects/Miuul_DataScientist_Bootcamp/Datasets/flo_data_20k.csv")
df = df_.copy()

# 2. Veri setinde
        # a. İlk 10 gözlem,
        # b. Değişken isimleri,
        # c. Boyut,
        # d. Betimsel istatistik,
        # e. Boş değer,
        # f. Değişken tipleri, incelemesi yapınız.

df.head(10)
df.columns
df.shape
df.describe().T
df.isnull().sum()
df.info()

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["omni_channel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["omni_channel_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

cols = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
df[cols] = df[cols].astype('datetime64[ns]')

#date_columns = df.columns[df.columns.str.contains("date")]
#df[data_columns] = df[date_columns].apply(pd.to_datetime, errors='coerce')

# df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)

# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakınız. 

df["order_channel"].unique()
df.groupby("order_channel")["master_id"].agg({"count"})
df["omni_channel"].mean()
df["omni_channel_total"].mean()


flo_describe = df.groupby("order_channel").agg({'master_id': lambda x: x.nunique(),
                                                'omni_channel': lambda x: x.sum(),
                                                'omni_channel_total': lambda x: x.sum()})


# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.sort_values(by='omni_channel_total', ascending=False).head(10)


# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.sort_values(by='omni_channel', ascending=False).head(10)


# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df

###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi

#df["last_order_date"].max()   #2021-05-30
#today_date = dt.datetime(2021, 6, 2)

today_date = df["last_order_date"].max() + dt.timedelta(days=2)
#today_date = dt.datetime(df["last_order_date"].max()) + dt.timedelta(days=2))


rfm = df.groupby("master_id").agg({'last_order_date': lambda date: (today_date - date.max()).days,
                                   'omni_channel': lambda x: x,
                                   'omni_channel_total': lambda num: num.sum()})

#(analysis_date - df["last_order_date"]).astype('timedelta64[D]')

rfm.columns = ["Recency", "Frequency", "Monetary"]
rfm.describe().T
rfm.sort_values(by='Monetary', ascending=False).head(10)

rfm_ = rfm.copy()
rfm_.index = rfm_.index.str[3:8]
print(rfm_.head())

rfm = rfm_display
rfm_display.unique()
rfm.head()

# customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe


###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi

rfm["Recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["Frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["Monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm["Recency_score"].astype(str) + rfm["Frequency_score"].astype(str))
rfm.head()

rfm[rfm["RF_SCORE"] == "55"].head()

# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi


###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme

seg_map = {
    r"[1-2][1-2]": 'hibernating',
    r"[1-2][3-4]": 'at_Risk',
    r"[1-2]5": 'cant_loose',
    r"3[1-2]": 'about_to_sleep',
    r"33": 'need_attention',
    r"[3-4][4-5]": 'loyal_customers',
    r"41": 'promising',
    r"51": 'new_customers',
    r"[4-5][2-3]": 'potential_loyalists',
    r"5[4-5]": 'champions'
}

rfm['Segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)


rfm[rfm["Segment"] == "cant_loose"].index


###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count"])


# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.


"champions","loyal_customers"
rfm[rfm["Segment"] == ("champions", "loyal_customers")].index


flo_ = rfm.merge(df, on="master_id")
flo_.info()

flo_.to_csv("OmniChannel.csv")

hedef = flo_[(flo_["Monetary"] > 250) & (flo_["interested_in_categories_12"].str.contains("KADIN", case=False, na=False)) & (flo_["Segment"].isin(["champions", "loyal_customers"]))]

hedef.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)



# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.

hedef2 = flo_[(flo_["interested_in_categories_12"].str.contains("ERKEK | COCUK", case=False, na=False)) & (flo_["Segment"].isin(["cant_loose", "about_to_sleep", "new_customers"]))]


##################################
#Tüm sürecin fonksiyonlaştırılması
##################################


import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("/Users/ecemuzman/PycharmProjects/Miuul_DataScientist_Bootcamp/.venv/Datasets/flo_data_20k.csv")
df = df_.copy()

def create_rfm (dataframe, csv= False ):

    # VERIYI HAZIRLAMA
    df["omni_channel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["omni_channel_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    cols = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    df[cols] = df[cols].astype('datetime64[ns]')

    # RFM METRIKLERININ HESAPLANMASI

    today_date = df["last_order_date"].max() + dt.timedelta(days=2)

    rfm = df.groupby("master_id").agg({'last_order_date': lambda date: (today_date - date.max()).days,
                                       'omni_channel': lambda x: x,
                                       'omni_channel_total': lambda num: num.sum()})

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    rfm["Recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["Frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["Monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["RF_SCORE"] = (rfm["Recency_score"].astype(str) + rfm["Frequency_score"].astype(str))

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['Segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)


    if csv:
        rfm.to_csv("rfm.csv")

    return rfm


df = df_.copy()

rfm_new = create_rfm(df, csv=True)
df[df["master_id"] == "00016786-2f5a-11ea-bb80-000d3a38a36f"]
