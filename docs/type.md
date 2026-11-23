## 🛠️ Построение модели через ручную реализацию (Линейная регрессия)

### Основные компоненты

- `Параметры (w, b):` тензоры с `requires_grad=True`, по ним считаются градиенты
- `Прямой проход** (forward):` ` X @ w + b` (для классификации поверх добавляется сигмоида)
- `Функция потерь:` считается вручную по формуле (MSE или BCE)
- `Обратный проход (backward):` `loss.backward()` считает градиенты автоматически
- `Шаг SGD:` `w_manual -= lr * w_manual.grad`, выполняется внутри `torch.no_grad()`
- `Обнуление градиентов:` `grad = None`, иначе градиенты накапливаются между итерациями

### Создание обучаемых параметров

``` python
import torch

w_manual = torch.randn(1, 1, requires_grad=true)
b_manual = torch.zeros(1, requires_grad=true)

lr = 0.01
n_epochs = 1000
```

### Ручной шаг обучения

``` python
for epoch in range(n_epochs):
  # Прямой проход
  y_pred = X_tensor @ w_manual + b_manual
  loss = ((y_pred - y_tensor)**2).mean()
  losses.append(loss.item())

  # Обратный проход: autograd считает градиенты
  loss.backward()

  # Обновление весов
  with torch.no_grad():
    w_manual -= lr* w_manual.grad
    b_manual -= lr* b_manual.grad

  # Обнуление весов
  w_manual.grad = None
  b_manual.grad = None
```

---

## ⚙️ Построение модели через готовые слои (Бинарная классификация)

### Основные компоненты

- **`nn.Linear`**: полносвязный слой, с входными признаками и выходными
- **`nn.Sigmoid`**: функция активации, от 0 до 1
- **`nn.ReLU`**: функция активации, от 0 до x

### Настройка нейросети

```python
import torch.nn as nn
import torch.optim as optim


# Модель: 2 входных признака -> 1 выход (вероятность от 0 до 1)
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# Функция потерь для бинарной классификации
criterion = nn.BCELoss()

# Оптимизатор SGD
optimizer = optim.SGD(model.parameters(), lr=1e-3)
```

### Готовые инструменты в обучении

```python
for epoch in range(n_epochs):
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    losses.append(loss.item())

    # Обнуление весов
    optimizer.zero_grad()

    # Обратный проход: autograd считает градиенты
    loss.backward()

    # Обновление весов
    optimizer.step()
```