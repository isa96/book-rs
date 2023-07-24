# -*- coding: utf-8 -*-
"""

##### Sumber Data : https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

Import library yang akan digunakan, yaitu NumPy untuk memproses operasi matematika yang akan digunakan pada dataframe dan Pandas untuk membangun dataframe dari file, dalam hal ini file .csv. Files digunakan untuk mengupload kaggle.json untuk menghubungkan Google Colab dengan Kaggle.
"""

import numpy as np
import pandas as pd

from google.colab import files

"""# Kaggle Setup"""

#install kaggle
!pip install -q kaggle

#unggah file json yang diunduh dari akun kaggle
uploaded = files.upload()

"""Buat direktori Kaggle dan pindahkan file yang diunggah ke folder baru. Kemudian berikan izin baca agar bisa diakses di Google Colab."""

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle

!chmod 600 /root/.kaggle/kaggle.json

"""## Mengunduh dan Menyiapkan Dataset

<img width="726" alt="image" src="https://user-images.githubusercontent.com/87566521/180254332-31f36886-46d7-4abb-94d7-a6fc1913bb65.png">

####**Informasi Dataset :**

Jenis | Informasi
--- | ---
Sumber | [Kaggle Dataset : Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
Lisensi | CC0: Public Domain
Kategori | Literature
Rating Pengunaan | 10.0 (Bronze)
Jenis dan Ukuran Berkas | CSV (106.94 MiB)
"""

!kaggle datasets download -d arashnic/book-recommendation-dataset

!unzip -q /content/book-recommendation-dataset.zip -d .

"""# Univariate Exploratory Data Analysis

Pada tahap ini, eksplorasi data dilakukan untuk memahami variabel-variabel dalam data dan korelasi antarvariabel. Eksplorasi dilakukan terhadap tiap 
file pada dataset yaitu Books.csv, Ratings.csv, dan Users.csv.
"""

# Memuat data pada sebuah dataframe menggunakan Pandas

Books = pd.read_csv('/content/Books.csv')
Ratings = pd.read_csv('/content/Ratings.csv')
Users = pd.read_csv('/content/Users.csv')

Books.head(10)

Books.info()

Books = Books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

Books.head(10)

print('Bentuk data (baris, kolom):'+ str(Books.shape))
print('Bentuk data (baris, kolom):'+ str(Ratings.shape))
print('Bentuk data (baris, kolom):'+ str(Users.shape))

"""### Variabel Books"""

# Memuat informasi dataframe
Books.info()

Books.describe()

# Melihat jumlah data kosong pada setiap kolom
Books.isnull().sum()

#Drop data yang kosong pada setiap kolom
Books = Books.dropna(axis=0)
Books.isnull().sum()

print('Total jumlah ISBN:', len(Books['ISBN'].unique()))
print('Total jumlah Book-Title :', len(Books['Book-Title'].unique()))
print('Total jumlah Publisher :', len(Books['Publisher'].unique()))

"""### Variabel Ratings"""

Ratings.head(5)

Ratings.info()

Ratings.describe()

"""Berdasarkan output di atas, dapat disimpulkan bahwa rating maksimum adalah 10 dan rating minimum adalah 0. Artinya, ada rating implisit yang ditunjukkan dengan nilai 0 yang selanjutnya akan kita anggap sebagai outlier karena skala rating yang sebenarnya berkisar antara 1 hingga 10. """

Ratings.isnull().sum()

print('Total Jumlah User :', len(Ratings['User-ID'].unique()))
print('Total Jumlah Rating :', len(Ratings['Book-Rating'].unique()))

"""## Variabel Users"""

Users.head(6)

Users.info()

Users.describe()

Users.isnull().sum()

# Drop data yang kosong pada setiap kolom
Users = Users.dropna(axis=0)
Users.info()

Users.isnull().sum()

print('Total Jumlah User :', len(Users['User-ID'].unique()))
print('Total Jumlah Lokasi :', len(Users['Location'].unique()))
print('Total Jumlah Umur:', len(Users['Age'].unique()))

"""# Data Preprocessing

Pada proses ini dilakukan penggabungan data pada variabel Books dan variabel Ratings dengan merge pada kondisi left yaitu menjaga semua baris dari dataframe pertama dan menambahkan kolom apa pun yang cocok berdasarkan kolom ISBN di dataframe kedua.
"""

book_rating = pd.merge(Books, Ratings, on='ISBN', how='left')

book_rating.head(5)

book_rating.isnull().sum()

"""Pada proses ini dilakukan penggabungan data pada variabel book_rating dan variabel Users dengan merge pada kondisi left yaitu menjaga semua baris dari dataframe pertama dan menambahkan kolom apa pun yang cocok berdasarkan kolom User-ID di dataframe kedua."""

books = pd.merge(book_rating, Users, on='User-ID', how='left')
books

books.head(2)

books = books[['User-ID','Age','Location','ISBN',	'Book-Title',	'Book-Author',	'Year-Of-Publication',	'Publisher',	'Book-Rating']]
books.head()

books.shape

"""# Data Preparation

Pada tahap ini dilakukan persiapan data dan beberapa teknik seperti mengatasi missing value dan menghapus NULL data.
"""

books.isnull().sum()

books = books.dropna()

books.isnull().sum()

books.shape

books.columns

"""Nama kolom di-rename sesuai keinginan, untuk lebih merepresentasikan kolom atau untuk memudahkan penulisan nama kolom pada tahapan selanjutnya."""

books.rename(columns={'User-ID':'userID','Book-Title':'Title', 'Book-Author':'Author','Year-Of-Publication':'PublicationYear','Book-Rating':'Rating'}, inplace=True)
books

"""Pada data preparation, akan diperiksa rating dengan nilai < 3 yang kemudian akan dihapus untuk kepentingan distribusi rating agar lebih merata."""

#  cek data yg memiliki rating <3
print('Jumlah rating < 3 :', books['Rating'].lt(3).sum())
books.shape

# menghilangkan data yang memiliki rating <3
books = books[books['Rating']>=3]
print('Jumlah total rating < 3 :', books['Rating'].lt(3).sum())
books.shape

"""Untuk mempercepat proses pemodelan, data difilter hanya dengan tahun 2003 dan 2004"""

# filter data dengan tahun 2003 dan 2004
books['PublicationYear'] = books['PublicationYear'].astype('str')
books = books[books['PublicationYear'].str.contains("2003|2004")]

"""Cek data duplikat pada kolom ISBN lalu bersihkan seluruh data duplikat"""

books_unique = books.drop_duplicates('ISBN')
books_unique

"""# Data Visualization

Visualisasi data bertujuan untuk mendapatkan insights dari dataset. Pada dataset ini, akan dilakukan visualisasi rating, top author, dan top publisher
"""

import seaborn as sns

with sns.axes_style('white'):
    g = sns.catplot('Rating', data=books, aspect=2.0, kind='count')
    g.set_ylabels('Total Rating')

import matplotlib.pyplot as plt

plt.subplots(figsize=(12,10))

ax=pd.Series(books_unique['Author']).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('muted',40))
for i, v in enumerate(pd.Series(books_unique['Author']).value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=10,color='white',weight='bold')
plt.title('Top Author')
plt.show()

plt.subplots(figsize=(12,10))

ax=pd.Series(books_unique['Publisher']).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('muted',40))
for i, v in enumerate(pd.Series(books_unique['Publisher']).value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=10,color='white',weight='bold')
plt.title('Top Publishers')
plt.show()

"""# Remove unused column

Drop kolom yang tidak digunakan pada pemodelan. Variabel books_unique akan digunakan pada tahapan implementasi KNN. Tujuan menggunakan variabel books_unique adalah untuk memperkecil jumlah data dan mempercepat proses pemodelan.
"""

books_unique=books_unique.drop('Age', axis=1)
books_unique=books_unique.drop('Location', axis=1)
books_unique

"""# KNN Implementation

## Creating Pivot

Pada tahapan ini, dibuat pivot tabel untuk mengkonversi tabel menjadi matriks 2D dan mengisi missing value dengan nol karena akan dilakukan 
perhitungan distance antara vektor rating. 

Pivot table dibuat dengan menggunakan variabel ratings dengan atribut userID, ISBN, dan Rating. Adapun tujuan menggunakan variabel baru ini adalah untuk memperkecil data agar proses pembuatan pivot lebih ringan dan aman pada CPU.
"""

ratings = books[['userID', 'ISBN', 'Rating']]
ratings

rating_pivot = ratings.pivot(index='ISBN', columns='userID', values='Rating').fillna(0)

"""## Model Development

Import libraries yang akan digunakan pada tahap pemodelan. Operator digunakan untuk menentukan atribut/kolom/elemen (data tuple) yang akan digunakan untuk sorting. Correlation digunakan untuk mendapatkan jarak antaritem. MAE digunakan untuk evaluasi model.
"""

import operator

from scipy.spatial.distance import correlation
from sklearn.metrics import mean_absolute_error as mae

"""Proses ini bertujuan untuk mendapatkan jarak (distance) antara item yang dipilih dengan semua item pada dataset"""

def get_distances(target_book):
    distances = []

    isbn = target_book['ISBN'].values[0]

    for index, book in books_unique.iterrows():
        if book['ISBN'] != isbn:
            dist = correlation(rating_pivot.loc[isbn].values, rating_pivot.loc[book['ISBN']].values)
            distances.append((book['ISBN'], book['Title'], dist))

    distances.sort(key=operator.itemgetter(2))
    return distances

"""### Finding the optimal K

Proses ini bertujuan untuk mendapatkan daftar rata-rata nilai MAE dengan melatih sebanyak 50 data secara acak dengan nilai K pada range yang ditentukan
"""

def get_error_rate(K_maks):
    result = []
    avg_error = 0

    for j in range(50):
        index = np.random.randint(len(books_unique))
        new_book = books_unique.iloc[index].to_frame().T
        distances = get_distances(new_book)

        total_rating = 0
        
        for i in range(K_maks):
            if j == 0:
                result.append([])

            neighbor = distances[i]
            current_book = books_unique[books_unique['ISBN'].str.contains(neighbor[0])].iloc[0]
            total_rating = total_rating + current_book[6]

            avg_rating = total_rating/(i+1)
            error = mae([new_book['Rating']], [avg_rating])

            result[i].append(error)

    for i in range(K_maks):
        result[i] = np.mean(result[i])

    return result

"""Proses ini bertujuan untuk mendapatkan nilai K optimal pada K = 11 - K = 173, dengan berdasarkan data latih sebanyak 50 data, dan nilai K optimal yang didapatkan adalah K = 15"""

import matplotlib.pyplot as plt

rate = 173

error_rate = get_error_rate(rate)
error_rate_range = error_rate[10:]

K = error_rate.index(min(error_rate_range))+1

plt.figure(figsize=(10,6))
plt.plot(range(11,rate+1), error_rate_range, color='blue', linestyle='dashed', 
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-", min(error_rate_range), "at K =", K)

"""<img width="394" alt="image" src="https://user-images.githubusercontent.com/87566521/180286544-d332d288-2ccf-48dd-b7e1-71aab15bf6f8.png">

# Predictor

Proses ini bertujuan untuk mendapatkan Top-10 Recommendations dan memprediksi rating, serta nilai MAE sesuai dengan nilai K optimal.
"""

def predict(query):
    new_book = books_unique[books_unique['Title'].str.contains(query)].iloc[0].to_frame().T
    print('Selected Book: ', new_book.Title.values[0])
    
    total_rating = 0
    distances = get_distances(new_book)
    
    print('\nTop 10 Recommended Books: \n')

    for i in range(K):
        neighbor = distances[i]
        current_book = books_unique[books_unique['ISBN'].str.contains(neighbor[0])].iloc[0]
        total_rating = total_rating + current_book[6] 
        if i < 10:
            print(current_book[2]+" | Author: "+str(current_book[3])+" | Rating: "+str(current_book[6]))

    print('\n')
    avg_rating = total_rating/K
    print('The predicted rating for %s is: %f' %(new_book['Title']. values[0], avg_rating))
    print('The actual rating for %s is: %f' %(new_book['Title']. values[0], new_book['Rating']))

    error = mae([new_book['Rating']], [avg_rating])

    # display
    print("Mean absolute error: " + str(error))

predict("Hard")

import matplotlib.pyplot as plt

rate = 179

error_rate = get_error_rate(rate)
error_rate_range = error_rate[10:]

K = error_rate.index(min(error_rate_range))+1

plt.figure(figsize=(10,6))
plt.plot(range(11,rate+1), error_rate_range, color='blue', linestyle='dashed', 
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-", min(error_rate_range), "at K =", K)

"""<img width="390" alt="image" src="https://user-images.githubusercontent.com/87566521/180286153-8d3f7f9f-0b2c-44eb-a127-cd7b8f737550.png">"""

def predict(query):
    new_book = books_unique[books_unique['Title'].str.contains(query)].iloc[0].to_frame().T
    print('Selected Book: ', new_book.Title.values[0])
    
    total_rating = 0
    distances = get_distances(new_book)
    
    print('\nTop 10 Recommended Books: \n')

    for i in range(K):
        neighbor = distances[i]
        current_book = books_unique[books_unique['ISBN'].str.contains(neighbor[0])].iloc[0]
        total_rating = total_rating + current_book[6] 
        if i < 10:
            print(current_book[2]+" | Author: "+str(current_book[3])+" | Rating: "+str(current_book[6]))

    print('\n')
    avg_rating = total_rating/K
    print('The predicted rating for %s is: %f' %(new_book['Title']. values[0], avg_rating))
    print('The actual rating for %s is: %f' %(new_book['Title']. values[0], new_book['Rating']))

    error = mae([new_book['Rating']], [avg_rating])

    # display
    print("Mean absolute error: " + str(error))

predict("Hard")

rate = 167

error_rate = get_error_rate(rate)
error_rate_range = error_rate[10:]

K = error_rate.index(min(error_rate_range))+1

plt.figure(figsize=(10,6))
plt.plot(range(11,rate+1), error_rate_range, color='blue', linestyle='dashed', 
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-", min(error_rate_range), "at K =", K)

"""<img width="393" alt="image" src="https://user-images.githubusercontent.com/87566521/180286358-f7bb9f87-41cc-4fcc-9d83-69f54872c43e.png">"""

def predict(query):
    new_book = books_unique[books_unique['Title'].str.contains(query)].iloc[0].to_frame().T
    print('Selected Book: ', new_book.Title.values[0])
    
    total_rating = 0
    distances = get_distances(new_book)
    
    print('\nTop 10 Recommended Books: \n')

    for i in range(K):
        neighbor = distances[i]
        current_book = books_unique[books_unique['ISBN'].str.contains(neighbor[0])].iloc[0]
        total_rating = total_rating + current_book[6] 
        if i < 10:
            print(current_book[2]+" | Author: "+str(current_book[3])+" | Rating: "+str(current_book[6]))

    print('\n')
    avg_rating = total_rating/K
    print('The predicted rating for %s is: %f' %(new_book['Title']. values[0], avg_rating))
    print('The actual rating for %s is: %f' %(new_book['Title']. values[0], new_book['Rating']))

    error = mae([new_book['Rating']], [avg_rating])

    # display
    print("Mean absolute error: " + str(error))

predict("Hard")

"""# Penutup

Mean Absolute Error (MAE) mengukur besarnya kesalahan pada prediksi rating terhadap data. Semakin rendah nilai MAE (Mean Absolute Error) maka semakin baik dan akurat model yang dibuat.

Berikut rumusnya :

![image](https://user-images.githubusercontent.com/87566521/139152819-30500f63-40a3-40ed-86fd-a62e517adbb4.png)

Hasil implementasi dari model yang telah dibukan akan digunakan untuk memberikan rekomendasi buku serta memprediksi rating yang akan diberikan oleh pengguna. Berikut adalah hasil implementasi Algoritma K-Nearest Neighbors pada sistem rekomendasi buku dengan judul buku “Hard Eight : A Stephanie Plum Novel (A Stephanie Plum Novel)”:


#### Nilai K dan Waktu Eksekusi
<img width="392" alt="image" src="https://user-images.githubusercontent.com/87566521/180284711-5a4698fb-6f28-4a27-a300-ffbc4f655e81.png">

#### Hasil Rekomendasi
<img width="545" alt="image" src="https://user-images.githubusercontent.com/87566521/180285055-d299ea27-d88b-466b-b045-30bd1401caf8.png">

#### Hasil Evaluasi MAE
<img width="432" alt="image" src="https://user-images.githubusercontent.com/87566521/180285373-65de6c08-1c37-4179-bc5d-ba8788ca4fab.png">
"""