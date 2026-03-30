# 🔢 Python e NumPy

Python se tornou a linguagem padrão para Machine Learning por um motivo simples: é fácil de escrever, tem uma comunidade enorme e um ecossistema de bibliotecas maduro. O **NumPy** é a base desse ecossistema — quase tudo em ML, por baixo dos panos, é uma operação sobre arrays NumPy.

---

## Listas vs. Arrays

Em Python puro, usamos listas para guardar coleções de valores. O problema é que listas não foram feitas para matemática:
```python
a = [1, 2, 3]
b = [4, 5, 6]

a + b        # [1, 2, 3, 4, 5, 6] — concatenou, não somou!
a * 2        # [1, 2, 3, 1, 2, 3] — repetiu, não multiplicou!
```

Com NumPy, operações matemáticas funcionam como esperado:
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b        # array([5, 7, 9])
a * 2        # array([2, 4, 6])
a ** 2       # array([1, 4, 9])
```

!!! tip "Por que isso importa para ML?"
    Um dataset é uma matriz NumPy. Cada linha é uma amostra, cada coluna é uma feature. Todas as operações do modelo (multiplicações, somas, médias) acontecem sobre essa estrutura.

---

## Operações Essenciais

**Criando arrays:**
```python
np.zeros((3, 4))       # matriz 3x4 de zeros
np.ones((2, 2))        # matriz 2x2 de uns
np.arange(0, 10, 2)    # array([0, 2, 4, 6, 8])
np.linspace(0, 1, 5)   # array([0, 0.25, 0.5, 0.75, 1.0])
```

**Formato e reshape:**
```python
a = np.arange(12)
a.shape          # (12,)

m = a.reshape(3, 4)
m.shape          # (3, 4) — 3 linhas, 4 colunas
```

**Slicing:**
```python
m[0]         # primeira linha
m[:, 1]      # segunda coluna inteira
m[1:3, 0:2]  # submatriz linhas 1-2, colunas 0-1
```

**Estatísticas básicas:**
```python
a = np.array([2, 4, 4, 4, 5, 5, 7, 9])

np.mean(a)    # 5.0
np.std(a)     # 2.0
np.min(a)     # 2
np.max(a)     # 9
```

---

## Produto Escalar

O produto escalar entre dois vetores é uma das operações mais usadas em ML, está no coração da regressão linear, das redes neurais e de vários outros algoritmos:
```python
pesos    = np.array([0.5, 0.3, 0.2])
features = np.array([10.0, 5.0, 2.0])

np.dot(pesos, features)   # 0.5*10 + 0.3*5 + 0.2*2 = 6.9
```

!!! note "O que isso representa?"
    Em regressão linear, a previsão é exatamente um produto escalar: cada feature multiplicada pelo seu peso, tudo somado. O modelo aprende **quais pesos usar**.

---

## Broadcasting

NumPy permite operar arrays de formatos diferentes sem precisar duplicar dados:
```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])

m + 10          # soma 10 a todos os elementos
m * np.array([1, 2, 3])   # multiplica cada coluna pelo valor correspondente
```

!!! warning "Cuidado com os shapes"
    Broadcasting tem regras precisas de compatibilidade de dimensões. Quando der erro de shape, verifique com `array.shape` antes de operar.