# DİLEKERZ NOTE
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #Verileri benzer ölçeklere getirmek için kullanıyoruz


# 1. IRIS VERİ SETİNİ YÜKLEME


iris = load_iris()

X = iris.data
y = iris.target

print("İlk 5 veri:")
print(X[:5])

print("\nİlk 5 sınıf:")
print(y[:5])#İlk 5 çiçeğin gerçek sınıfını gösterir



# 2. VERİLERİ EĞİTİM VE TEST OLARAK AYIRMA
 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20, #Verinin %20si test için kullanılacak
    random_state=42,
    stratify=y #Iris veri setinde 3 sınıf var ,stratify=y sayesinde eğitim ve test kümelerinde sınıfların oranı korunur
)



# 3. VERİLERİ ÖLÇEKLENDİRME


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train) #Eğitim verisinin:ortalamasını standart sapmasını Sonra verileri ölçeklendiriyor
X_test = scaler.transform(X_test) # Eğitim verisinden öğrenilen ölçekleme kuralları test verisine uygulanıyor




# 4. NUMPY VERİLERİNİ PYTORCH TENSOR'E ÇEVİRME

X_train = torch.tensor(X_train, dtype=torch.float32) #Eğitim girişlerini PyTorch tensorüne dönüştürüyoruz
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long) #dtype=torch.long kullanmamızın nedeni CrossEntropyLoss fonksiyonunun sınıf numaralarını integer formatında istemesidir


# 5. NORMAL NEURAL NETWORK MODELİ

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = NeuralNetwork()


# 6. LOSS VE OPTIMIZER

criterion = nn.CrossEntropyLoss() #Modelin yaptığı tahmin ile gerçek cevap arasındaki hatayı hesaplıyor

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01 #Learning Rate yani öğrenme oranıdır AĞırlıkların her eğitim adımında ne kadar değişeceğini etkiler
)


# 7. NORMAL MODELİ EĞİTME

epochs = 300

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward() #Hatadan hangi ağırlık ne kadar sorumlu?sorusunu hesaplıyor

    optimizer.step()#Ağırlıkları güncelleme

    if (epoch + 1) % 50 == 0:
#Her 50 epoch'ta sonucu gösteriyoruz
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}"
            #Epoch 50/300 - Loss: 0.3821 mesela 
        )


# 8. NORMAL MODELİN DOĞRULUĞUNU HESAPLAMA

with torch.no_grad(): #Bu noktada artık eğitim yapmıyoruz torch.no_grad(): Gradient hesaplama

    outputs = model(X_test) #Model daha önce eğitimde görmediği test verileri üzerinde tahmin yapıyor

    predictions = torch.argmax(outputs, dim=1) #Tahmin edilen sınıfı seçiyoruz,argmax en büyük değerin indeksini bulur

    correct = (predictions == y_test).sum().item() #Doğru tahminleri sayıyoruz

    normal_accuracy = correct / len(y_test) * 100 #Doğru tahmin sayısını toplam test verisine bölüyoruz.


print("\nNormal Neural Network Accuracy:")
print(f"%{normal_accuracy:.2f}")


# 9. TERNARY QUANTIZATION FONKSİYONU

def quantize_weights(weights, threshold_ratio=0.15):
#Ağırlıkları ternary hale getiren fonksiyon oluşturuyoruz
    max_weight = torch.max(torch.abs(weights)) #En büyük ağırlığı buluyoruz
    #torch.abs(weights):ağırlıkların mutlak değerini alır

    delta = threshold_ratio * max_weight # eşik değerimizdir

# ternary dönüşüm:
    ternary_weights = torch.where(
        weights > delta,
        torch.ones_like(weights),
        torch.where(
            weights < -delta,
            -torch.ones_like(weights),
            torch.zeros_like(weights)
        )
    )
# torch.ones_like(weights),+1 yapdemek
    return ternary_weights


# 10. NORMAL MODELİN AĞIRLIKLARINI GÖSTERME

print("\nNormal fc1 ağırlıkları:")

print(model.fc1.weight.data)


# 11. AĞIRLIKLARI TERNARY HALE GETİRME

with torch.no_grad(): #torch.no_grad() tekrar kullanılıyor burada ağırlıkları elle değiştireceğimiz için PyTorchun gradient sisteminin bunu takip etmesini istemiyoruz

    model.fc1.weight.data = quantize_weights(
        model.fc1.weight.data
    )

    model.fc2.weight.data = quantize_weights(
        model.fc2.weight.data
    )


print("\nTernary fc1 ağırlıkları:")

print(model.fc1.weight.data)


# 12. TERNARY MODELİN DOĞRULUĞUNU HESAPLAMA

with torch.no_grad(): #tekrar gradient hesaplamıyoruz çünkü eğitim yapmıyoruz

    outputs = model(X_test) #ağırlıkları artık ternary

    predictions = torch.argmax(outputs, dim=1)

    correct = (predictions == y_test).sum().item()

    ternary_accuracy = correct / len(y_test) * 100


print("\nTernary Neural Network Accuracy:")
print(f"%{ternary_accuracy:.2f}")


# 13. İKİ MODELİ KARŞILAŞTIRMA

print("\n-------------------------------")
print("SONUÇLAR")
print("-------------------------------")

print(
    f"Normal Neural Network : %{normal_accuracy:.2f}"
)

print(
    f"Ternary Neural Network: %{ternary_accuracy:.2f}"
)