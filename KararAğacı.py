 # csv dosyalarını okumak için
import pandas as pd

# csv dosyamızı okuduk.
data = pd.read_csv('C:/Users/Administrator/Desktop/maaslar.csv')

# egitim kolonunun değerlerini bir değişkene atadık
egitim = data.iloc[:, 1:2].values

# maas kolonunun değerlerini bir değişkene atadık
maas = data.iloc[:, 2:3].values

# DecisionTreeRegressor sınıfını import ettik
from sklearn.tree import DecisionTreeRegressor

# DecisionTreeRegressor sınıfından bir nesne ürettik
# random_state modelin çıkışını çoğaltılamaz hale getirir yani random_state değeri belli olduğunda aynı parametreler ve aynı eğitim verisi verilmişse, aynı sonuçlar üretecektir.

dtr = DecisionTreeRegressor(random_state=0)

# Makinemizi eğittik
dtr.fit(egitim, maas)

# Decision Tree algoritmasını kullanarak eğittiğimiz makinenin egitim değerlerine göre bir tahmin yapmasını sağlıyoruz.
predict = dtr.predict(egitim)

# grafik çizmek için
import matplotlib.pyplot as plt

# Grafik şeklinde ekrana basmak için
plt.scatter(egitim, maas, color='red')
plt.plot(egitim, predict, color='blue')
plt.show()