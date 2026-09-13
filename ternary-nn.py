#DİLEKERZ NOTE: quantization mantığı aslına indirgeme işlemidir, sinir ağındaki ağırlıkları -1,0,1 dğerlerine döndürü
weights = [0.8, -0.5, 0.1, -0.05, 0.9]

threshold = 0.2 #EŞİK değerimiz

ternary_weights = []

for w in weights:

    if w > threshold:
        ternary_weights.append(1)

    elif w < -threshold:
        ternary_weights.append(-1)

    else:
        ternary_weights.append(0)

print("Normal ağırlıklar:")
print(weights)

print("Ternary ağırlıklar:")
print(ternary_weights)

print("***---------------------------------------------***")

#  gerçek yapay sinir ağlarında sinir apğırlıklarını pythone listesi olarak değil ,tensor olarak tutuluyor
import torch

weights = torch.tensor([
    0.8,
    -0.5,
    0.1,
    -0.05,
    0.9
])

threshold = 0.2

ternary_weights = torch.where(
    weights > threshold,
    torch.tensor(1.0),
    torch.where(
        weights < -threshold,
        torch.tensor(-1.0),
        torch.tensor(0.0)
    )
)

print("Normal ağırlıklar:")
print(weights)

print("\nTernary ağırlıklar:")
print(ternary_weights)

print("***---------------------------------------------***")

# şimdi Bunu bir quantize() fonksiyonuna çevirelim:
def quantize(weights, threshold=0.2):

    ternary_weights = torch.where(
        weights > threshold,
        torch.tensor(1.0),
        torch.where(
            weights < -threshold,
            torch.tensor(-1.0),
            torch.tensor(0.0)
        )
    )

    return ternary_weights


weights = torch.tensor([
    0.8,
    -0.5,
    0.1,
    -0.05,
    0.9
])

result = quantize(weights)

print("Normal:")
print(weights)

print("Ternary:")
print(result)

print("***---------------------------------------------***")
print("deltayı hesapladıktan sonraki ağırlık kıyaslamaları:")

weights = torch.tensor([
    0.8,
    -0.5,
    0.1,
    -0.05,
    0.9
])

THRESHOLD = 0.15

delta = THRESHOLD * torch.max(torch.abs(weights))

ternary_weights = torch.where(
    weights > delta,
    torch.tensor(1.0),
    torch.where(
        weights < -delta,
        torch.tensor(-1.0),
        torch.tensor(0.0)
    )
)

print("Normal ağırlıklar:")
print(weights)

print("\nDelta:")
print(delta)

print("\nTernary ağırlıklar:")
print(ternary_weights)

print("***---------------------------------------------***")
print("normal bir ağ oluşturuyoruz yani :GERÇEK NEURAL NETWORK")

import torch.nn as nn

class NormalNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = NormalNetwork()

print(model)
print(model.fc1.weight)   # bu full-precision weights yani normal ağırlıklarımız

# Şunları ternary yapabiliriz: 
print("sinir ağı katmanının ağırlıklarını ternary hale getirmiş oluyoruz:")
weights = model.fc1.weight.data

delta = 0.15 * torch.max(torch.abs(weights))

ternary_weights = torch.where(
    weights > delta,
    torch.tensor(1.0),
    torch.where(
        weights < -delta,
        torch.tensor(-1.0),
        torch.tensor(0.0)
    )
)

print(ternary_weights)