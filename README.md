## Необходимые библиотеки для работы в Python

Пакеты для конкретной версии Python
```
pip install torch torchvision numpy pandas matplotlib scikit-learn
```
Или (Windows и VS Code)

```
py -m pip install torch torchvision numpy pandas matplotlib scikit-learn
```

---


# Решение задач с помощью машинного обучения


| Тип задачи | Примеры | Функция потерь | Оптимизатор |
|---|---|---|---|
| 1. [**Регрессия**](pytorch_tasks/task1_regression.ipynb) | цена квартиры, температура, зарплата | `MSELoss` (MSE) | SGD |
| 2. [**Бинарная классификация**](pytorch_tasks/task2_classification.ipynb) (2 класса) | кошка или собака | `BCELoss` (BCE) | SGD |
| 3. [**Многоклассовая классификация (MLP)**](pytorch_tasks/task3_multiclassification.ipynb) (3+ класса) | цифры 0–9, кошка / собака / птица | `CrossEntropyLoss` | Adam или SGD |

---
## ✅ Готовое решение (.ipynb)
[1. Регрессия ](pytorch_tasks/task1_regression.ipynb)

[2. Классификация (2 класса)](pytorch_tasks/task2_classification.ipynb)

[3. Многоклассовая классификация (числа от 0 до 9)](pytorch_tasks/task3_multiclassification.ipynb)

[4. Свёртка](pytorch_tasks/task4_conv.ipynb)

---
## 📄 Документация 
[1. Ответы на билеты (теория)](docs/answers.md)

[2. Теория для изучения](docs/guide.md)

[3. Словарь терминов](docs/terms.md)

[4. Реализация ручного и готового обучения модели](docs/type.md)

---

## 💡 Заметки
Раздел будет дополняться.


