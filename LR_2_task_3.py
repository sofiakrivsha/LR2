import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
# 1. Визначаємо назви стовпців та завантажуємо дані 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
# 2. Вивід інформації для перевірки 
print("Форма датасету:", dataset.shape)
print(dataset.head(20))
# 3. Побудова графіків (Крок 2) 
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# 4. Побудова гістограм 
dataset.hist()
plt.show()
# 5. Матриця розсіювання 
scatter_matrix(dataset)
plt.show()
