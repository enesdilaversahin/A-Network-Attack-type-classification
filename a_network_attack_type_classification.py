# -*- coding: utf-8 -*-

# Gerekli kütüphaneleri yükleyin
!pip install scikit-learn pandas matplotlib seaborn --quiet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# CSV formatındaki veri setini yükleyin
file_path = '/content/drive/My Drive/kddcup.data.corrected'  # Dosyanızın yolu
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
           "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
           "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
           "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
           "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
           "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
           "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack",
           "last_flag"]

data = pd.read_csv(file_path, header=None, names=columns)
data.head()

pd.set_option('display.max_rows', 10)

if 'data' in globals():
    print("data değişkeni tanımlı.")
else:
    print("data değişkeni tanımlı değil.")

# Veri keşfi
print("Veri Seti Boyutları:", data.shape)
print("Eksik Değerler:\n", data.isnull().sum())

# Saldırı türlerine göre dağılım
print("\nSaldırı Türleri Dağılımı:\n", data['attack'].value_counts())

# Özet istatistikler
data.describe()

# Saldırı türlerine göre dağılımı görselleştirme
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='attack', order=data['attack'].value_counts().index)
plt.title("Saldırı Türlerine Göre Dağılım")
plt.xlabel("Saldırı Türü")
plt.ylabel("Frekans")
plt.xticks(rotation=90)
plt.show()

print(data['attack'].unique())


# İkili sınıflandırma hedef değişkeni
data['binary_label'] = data['attack'].apply(lambda x: 0 if x == "normal" else 1)

# Çoklu sınıflandırma hedef değişkeni
attack_mapping = {
    "normal.": "normal",  # Normal traffic
    "neptune.": "dos",    # DoS attack
    "smurf.": "dos",      # DoS attack
    "teardrop.": "dos",   # DoS attack
    "land.": "dos",       # DoS attack
    "back.": "dos",       # DoS attack (added based on context)
    "pod.": "dos",        # DoS attack (added based on context)
    "satan.": "probe",    # Probing/scanning attack
    "ipsweep.": "probe",  # Probing/scanning attack
    "portsweep.": "probe",# Probing/scanning attack
    "nmap.": "probe",     # Probing/scanning attack
    "guess_passwd.": "r2l",  # Remote-to-local attack
    "ftp_write.": "r2l",     # Remote-to-local attack
    "imap.": "r2l",          # Remote-to-local attack
    "phf.": "r2l",           # Remote-to-local attack
    "multihop.": "r2l",      # Remote-to-local attack
    "warezmaster.": "r2l",   # Remote-to-local attack
    "warezclient.": "r2l",   # Remote-to-local attack
    "spy.": "r2l",           # Remote-to-local attack
    "buffer_overflow.": "u2r",  # User-to-root attack
    "loadmodule.": "u2r",       # User-to-root attack
    "perl.": "u2r",             # User-to-root attack
    "rootkit.": "u2r",          # User-to-root attack
}
data['multi_label'] = data['attack'].map(attack_mapping)

# Sonuçları kontrol et
print("İkili Sınıflandırma Etiketleri:\n", data['binary_label'].value_counts())
print("\nÇoklu Sınıflandırma Etiketleri:\n", data['multi_label'].value_counts())



print(data['attack'].isnull().sum())  # Eksik değerleri kontrol et
data['attack'] = data['attack'].fillna("normal")  # Eksik değerleri "normal" ile doldur

if 'data' in globals():
    print("data değişkeni tanımlı.")
else:
    print("data değişkeni tanımlı değil.")

# Kategorik değişkenleri kodlama
data_encoded = pd.get_dummies(data, columns=["protocol_type", "service", "flag"], drop_first=True)

# Özelliklerden gereksiz sütunları çıkar
X = data_encoded.drop(['attack', 'binary_label', 'multi_label'], axis=1)

# İkili sınıflandırma için hedef değişken
y_binary = data['binary_label']

# Çoklu sınıflandırma için hedef değişken
y_multi = data['multi_label']

# Eğitim ve test verisi
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.5, random_state=42)

# Random Forest Modeli
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Tahmin
y_pred = rf_model.predict(X_test)

# Performans değerlendirme
print("İkili Sınıflandırma Sonuçları:\n", classification_report(y_test, y_pred))

if 'data' in globals():
    print("data değişkeni tanımlı.")
else:
    print("data değişkeni tanımlı değil.")


# Hedef Değişken
y_multi = data["multi_label"]

# Eğitim ve Test Verisi
X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.5, random_state=42)

# Model Eğitimi
rf_multi_model = RandomForestClassifier()
rf_multi_model.fit(X_train, y_train)

# Tahmin ve Değerlendirme
y_multi_pred = rf_multi_model.predict(X_test)
print(classification_report(y_test, y_multi_pred))

# Daha güzel ve anlaşılır bir görselleştirme için stil ve renk paletini değiştirebiliriz
plt.figure(figsize=(10,6))
sns.set(style="whitegrid")  # Beyaz grid arka plan

# Saldırı türlerine göre dağılım
ax = sns.countplot(data=data, x="multi_label", order=data['multi_label'].value_counts().index)

# Başlık, etiketler ve estetik düzenlemeler
plt.title("Saldırı Türlerine Göre Dağılım", fontsize=16, weight='bold')
plt.xlabel("Saldırı Türü", fontsize=12)
plt.ylabel("Frekans", fontsize=12)

# X eksenindeki etiketleri döndürme (yazılar kesilmesin diye)
plt.xticks(rotation=45, ha='right')

# Her çubuğun üstüne frekans değeri eklemek
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

# Grafiği gösterme
plt.tight_layout()  # Grafik sıkışmasın diye düzenleme yapar
plt.show()