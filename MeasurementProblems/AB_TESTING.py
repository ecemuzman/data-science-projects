#####################################################
# AB Testi ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################
import pandas as pd

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchase metriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsx excel’inin ayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.



import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

control = pd.read_excel("Datasets/ab_testing.xlsx", sheet_name='Control Group')
test = pd.read_excel("Datasets/ab_testing.xlsx", sheet_name='Test Group')
control.head()
test.head()
# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

control.describe()  #Maximum Bidding
test.describe()   #Average Bidding

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

control["Group"] = "control"
test["Group"] = "test"
df = pd.concat([control, test], axis=0)
df.info()

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

#Control ve Test gruplarına uygulanan farklı bidding türleri ile Purchase ortalamaları arasında anlamlı bir fark var mı ?

# H0: M1 = M2  Control ve Test gruplarının “Purchase” ortalamaları arasında fark yoktur.
# H1: M1 != M2  fark vardır.

# Adım 2: Kontrol ve test grubu için purchase ortalamalarını analiz ediniz

df.groupby("Group")["Purchase"].mean()

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

test_stat, pvalue = shapiro(df.loc[df["Group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

#H0 reddedilemez. Normal dağılım varsayımı sağlanmaktadır.

# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["Group"] == "control", "Purchase"],
                           df.loc[df["Group"] == "test", "Purchase"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0 reddedilemez. varyanslar homojendir.

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

test_stat, pvalue = ttest_ind(df.loc[df["Group"] == "control", "Purchase"],
                              df.loc[df["Group"] == "test", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# p-value < değilse 0.05 H0 REDDEDILEMEZ.
# p-value = 0.3493
# H0 reddedilemez. o zaman istatistiki olarak anlamlı bir fark yoktur.


##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# ttest kullandım. çünkü varsayım dağılımı (shapiro) ve varsayım homojenliğine (levenue) baktığımızda
# p-valuelar 0.05 'ten yüksek çıkıyor varsayımlar sağlanıyor. bu da bize parametrik test uygulamamız gerektiğini söylüyor.
# ttest uyguladığımızda ise anlamlı bir fark olmadığını gösteriyor.


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.


# İki farklı bidding uygulamasının daha iyi ya da daha kötü olmadığı anlamına gelir.
# Veri arttırılarak denenebilir.
# Diğer değişkenler ile denenebilir.
# Farklı segmentlere bakılarak denenebilir.

df.groupby("Group")["Earning"].mean()


test_stat, pvalue = shapiro(df.loc[df["Group"] == "control", "Earning"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Group"] == "test", "Earning"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# her iki grubun dağılımı normaldir.

test_stat, pvalue = levene(df.loc[df["Group"] == "control", "Earning"],
                           df.loc[df["Group"] == "test", "Earning"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# varyanslar homojendir

test_stat, pvalue = ttest_ind(df.loc[df["Group"] == "control", "Earning"],
                              df.loc[df["Group"] == "test", "Earning"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05
# H0 reddedilir. anlamlı olarak fark vardır.