
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################


# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

movie = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/movie.csv')

movie.head()

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti

rating = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/rating.csv')


# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
df_ = movie.merge(rating, how="left", on="movieId")
df = df_.iloc[:len(df_) // 2]   # cok buyuk bir veri seti oldugu icin kolay calismak adina satırlar yariya dusurulebilir
df.shape
df.head()
df = df_.copy()

# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.

df["title"].value_counts().head(20)

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz

comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.columns = ["title"]
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
df = df[~df["title"].isin(rare_movies)]
df.shape

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.

user_movie_df = df.pivot_table(index="userId", columns="title", values="rating")

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım


def create_user_movie_df():
    import pandas as pd

    movie = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/rating.csv')

    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    comment_counts.columns = ["title"]
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

user_movie_df.head()

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.

random_user = df["userId"].dropna().astype(int).sample(1, random_state=45).values[0]


# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

random_user_df = df[df["userId"] == random_user]


# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.

movies_watched = random_user_df["title"].tolist()

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.

movies_watched_df = user_movie_df[movies_watched]


# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.

user_movie_count = movies_watched_df.T.notnull().sum().reset_index()

user_movie_count.columns = ["userId", "movie_count"]

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

perc = len(movies_watched) * 60 / 100

users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

users_same_movies.reset_index()
users_same_movies.columns = ["userId", "movie_count"]

movies_watched_df = user_movie_df[movies_watched]

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.


final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

""" final_df = final_df.T.drop_duplicates().T
final_df = final_df[~final_df.index.duplicated(keep='first')]

"""
final_df = final_df.drop_duplicates()

corr_df = final_df.T.corr().unstack().sort_values()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

corr_df[corr_df["user_id_1"] == random_user]



# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.


top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users


# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz


rating = pd.read_csv('/Users/ecemuzman/PycharmProjects/Miuul_DataScientist_Bootcamp/Recommendation Systems/datasets/movie_lens_dataset/rating.csv')

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

top_users_ratings


#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]

# Adım 2: Film id’sivle her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.

recommendation_df = (top_users_ratings.groupby("movieId")["weighted_rating"].mean().reset_index(name="weighted_rating_mean"))

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.

movies_to_be_recommend = (recommendation_df.loc[recommendation_df["weighted_rating_mean"] > 3.5].sort_values("weighted_rating_mean", ascending=False))

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.


movies_to_be_recommend.head(5)

#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

# Adım 1: movie,rating veri setlerini okutunuz.

movie = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/rating.csv')

df = movie.merge(rating, how="left", on="movieId")
# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.

df.head()

df["userId"].isna().sum()
df = df.dropna(subset=["userId"])
df["userId"] = df["userId"].astype("int64")
df["rating"] = df["rating"].astype("int64")

movie_id = (df[(df["userId"] == user) & (df["rating"] == 5)]).sort_values("timestamp", ascending=False).iloc[0]["movieId"]


# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.

movie_title = movie.loc[movie["movieId"] == movie_id, "title"].values[0]

selected_movie_df = user_movie_df[movie_title]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.

item_corr = user_movie_df.corrwith(selected_movie_df)

item_corr.dropna().sort_values(ascending=False)

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’i öneri olarak veriniz.

recommended_movies = (item_corr.dropna().sort_values(ascending=False).iloc[1:6])




