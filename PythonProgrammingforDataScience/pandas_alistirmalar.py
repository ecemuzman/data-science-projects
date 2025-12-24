import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

#Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
df = sns.load_dataset("titanic")
df.columns
df.describe().T
df.index
df.info()

#Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
df["sex"].value_counts()

#Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
df.nunique()
df.columns.nunique()

#Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.
df["pclass"].nunique()

#Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
df[["pclass","parch"]].nunique()

#Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"]

#Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
df[df["embarked"] == "C"].head()

#Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df[df["embarked"] != "S"].head()

#Görev9: Yaşı 30dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

df[(df["age"] < 30) & (df["sex"] == "female")]

#Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.

df[(df["fare"] > 500) | (df["age"] > 70)]

#Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
df.isnull().sum()

#Görev 12: who değişkenini dataframe’den çıkarınız.
df = df.drop("who", axis=1, inplace=True)
df

#Görev 13: deck değişkenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
df["deck"] = df["deck"].fillna(df["deck"].mode()[0])

#Görev 14: age değişkenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df["age"] = df["age"].fillna(df["age"].median())

#Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})

#Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın.
#Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz.
# (apply ve lambda yapılarını kullanınız)

df['age_flag'] = df['age'].apply(lambda x: 0 if x >= 30 else 1)
df.head()

########################
#Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

import seaborn as sns
df = sns.load_dataset("tips")
df.columns
df.head()

#Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
df.groupby(["time"]).agg({"total_bill" : ["sum", "min", "max", "mean"]})

#Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

#Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],"tip": ["sum", "min", "max", "mean"]})

#Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
n_df = df.loc[(df['size'] < 3) & (df['total_bill'] > 10), "total_bill"].mean()
n_df

#Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

#Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
df_sorted = df.sort_values(by='total_bill_tip_sum', ascending=False).head(30) #[:30]

df_sorted.shape


