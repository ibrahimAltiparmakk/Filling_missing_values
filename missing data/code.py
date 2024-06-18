# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:44:26 2024

@author: sarib
"""

# as kısmı kütüphaneyi nasıl isimlendireceğimizi belirtir

import pandas as pd  #veri işlemlerinde kullanılır
import numpy as np  #büyük sayılar ya da hesaplama işlemleri için kullandığımız kütüphane

#kodlar

# veri yükleme

#pandas kütüphanesini verileri okumak için kullanıyoruz
#veri dosyamız csv dosyası olduğu için read_csv diyoruz
veriler = pd.read_csv("eksikveriler.csv")

#---------------------------------------------------

#eksik veriler
from sklearn.impute import SimpleImputer
#Kısaca, SimpleImputer eksik verilerinizi otomatik olarak belirlediğiniz stratejiye göre doldurarak analizlerinizde veya modellemelerinizde daha temiz bir veri seti ile çalışmanızı sağlar.

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
#strategy eksik değerleri nasıl bir stratejiyle dolduracağımızı söyleriz , burada mean yani sutun un  ortalamasınıyla dolduracağımız anlamına gelir

Yas = veriler.iloc[:,1:4].values 
#iloc metodu,pandas kütüphanesinden gelir belirli satır ve sütunları seçmek için sayısal indeksleri kullanmanıza olanak tanır , virgülüun solundaki kısım satırları gösterir eğer o kısıma hiç bir şey yazmazsak bütün satırları alır , virgülün sağındaki kısımda ise sutun kısmını belirleriz 1:4 demek sütunların 1'den 3'e kadar olanlarını (1:4) seçer indeksleme 0 dan başladığı için 1:4 ifadesi 1, 2 ve 3 numaralı sütunları seçer 

print(Yas)

imputer = imputer.fit(Yas[:,1:4]) #fit ile belirlediğimiz sutundaki değerleri öğretiriz
Yas[:,1:4]=imputer.transform(Yas[:,1:4])#transform ile de öğrendiğini uygulamasını söylüyoruz

print(Yas)
        
#-------------------------------------------------------
    
        
        
        