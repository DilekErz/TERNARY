# Ternary Neural Network (TNN)

Bu proje, yazılım geliştirme stajım kapsamında **Ternary Neural Network (TNN)**  ve **weight quantization**  konularını öğrenmek ve uygulamak amacıyla geliştirilmiştir.
Çalışmada ilk olarak sinir ağı ağırlıklarının `-1`, `0`ve `+1` değerlerine dönüştürülmesi incelenmiş, ardından PyTorch kullanılarak bu dönüşüm gerçek bir yapay sinir ağı üzerinde uygulanmıştır. Son aşamada Iris veri seti kullanılarak normal Neural Network modeli ile ternary ağırlıklara dönüştürülmüş modelin doğruluk sonuçları karşılaştırılmıştır.

## Projenin Amacı

Ternary Neural Network yapısında normalde ondalıklı değerlerden oluşan sinir ağı ağırlıkları üç farklı değere indirgenir:

- `1`
- `0`
- `+1`

Bu projede temel amaç, bu dönüşümün çalışma mantığını öğrenmek ve ağırlıkların ternary hale getirilmesinin model performansı üzerindeki etkisini gözlemlemektir.

## Ternary Quantization

Ağırlıkların ternary değerlere dönüştürülmesi için bir eşik değeri kullanılmaktadır.

- weight > delta   → +1
- weight < -delta  → -1
- diğer durumlar   → 0

Projede dinamik eşik değeri aşağıdaki şekilde hesaplanmıştır:

`delta = threshold_ratio * torch.max(torch.abs(weights))`

Ardından PyTorch kullanılarak ağırlıklar ternary değerlere dönüştürülmüştür:

def quantize_weights(weights, threshold_ratio=0.15):
    max_weight = torch.max(torch.abs(weights))
    delta = threshold_ratio * max_weight

    ternary_weights = torch.where(
        weights > delta,
        torch.ones_like(weights),
        torch.where(
            weights < -delta,
            -torch.ones_like(weights),
            torch.zeros_like(weights)
        )
    )

    return ternary_weights
## Kullanılan Veri Seti

Model uygulamasında Scikit-learn içerisinde bulunan **Iris veri seti** kullanılmıştır.
Veri setinde üç farklı çiçek sınıfı bulunmaktadır:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Veriler `%80` eğitim ve `%20` test olacak şekilde ayrılmış, sınıf dağılımının korunması için `stratify=y` kullanılmıştır. Giriş verileri ayrıca `StandardScaler` ile ölçeklendirilmiştir.

## Model Mimarisi

PyTorch kullanılarak basit bir fully connected Neural Network oluşturulmuştur.

`4 Giriş`
↓  
`8 Nöronlu Gizli Katman`
↓  
`ReLU`
↓  
`3 Çıkış`

Model:

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

Model **300 epoch**  boyunca eğitilmiştir. Kayıp fonksiyonu olarak `CrossEntropyLoss`, optimizer olarak ise `Adam` ve `0.01` learning rate kullanılmıştır.

## Normal ve Ternary Model Karşılaştırması

İlk olarak model normal floating-point ağırlıklarla eğitilmiş ve test doğruluğu hesaplanmıştır.
Daha sonra eğitilmiş modelin `fc1` ve `fc2` katmanlarındaki ağırlıklar ternary değerlere dönüştürülmüştür:

with torch.no_grad():
    model.fc1.weight.data = quantize_weights(model.fc1.weight.data)
    model.fc2.weight.data = quantize_weights(model.fc2.weight.data)

Ternary dönüşümden sonra model aynı test verileri üzerinde tekrar çalıştırılmış ve iki modelin doğruluk değerleri karşılaştırılmıştır.

-------------------------------
SONUÇLAR
-------------------------------

Normal Neural Network : %96.67
Ternary Neural Network: %90.00

Bu karşılaştırma ile ternary quantization işleminin model doğruluğu üzerindeki etkisi gözlemlenmiştir.

## Proje Dosyaları
- ternary-nn.py — Ternary quantization mantığı, threshold/delta hesaplama, Tensor işlemleri ve temel Neural Network ağırlık dönüşümlerini içerir.
- ternary-nn2.py — Iris veri seti üzerinde Neural Network eğitimi, ternary quantization ve normal/ternary model karşılaştırmasını içerir.
## Kullanılan Teknolojiler
- Python
- PyTorch
- Scikit-learn
- NumPy
- Git & GitHub
## Öğrenilen Konular
- Ternary Neural Network (TNN)
- Weight Quantization
- Full-Precision Weights
- Threshold ve Delta
- PyTorch Tensor işlemleri
- Artificial Neural Networks
- ReLU
- CrossEntropyLoss
- Adam Optimizer
- Backpropagation
- Train/Test Split
- Feature Scaling
- Model Accuracy
## Sonuç

Bu çalışma ile **Ternary Neural Network**  ve **weight quantization**  yapısının temel çalışma mantığı uygulamalı olarak incelenmiştir. Normal floating-point ağırlıkların `-1`, `0` ve `+1` değerlerine dönüştürülmesi gerçekleştirilmiş ve bu dönüşüm Iris veri seti üzerinde eğitilen bir Neural Network modeline uygulanmıştır.
Normal ve ternary model sonuçları karşılaştırılarak quantization işleminin model performansı üzerindeki etkisi gözlemlenmiştir.
Bu proje, yazılım geliştirme stajı kapsamında Ternary Neural Network ve model quantization konularında yapılan teorik ve uygulamalı çalışmaların bir parçasıdır.

## Geliştirici
**Dilek Ayça Ersöz** 
