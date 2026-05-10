# Крок 1 та 2 код
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
# Крок 3 та 4 (Порівняння алгоритмів) код
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# 1. Розділення даних на навчальну та контрольну вибірки (80/20) 
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# 2. Підготовка списку алгоритмів для перевірки [cite: 358-365]
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# 3. Оцінка кожної моделі по черзі 
results = []
names_models = []
for name, model in models:
    # Використовуємо 10-кратну крос-валідацію 
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names_models.append(name)
    # Вивід середньої точності та стандартного відхилення 
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# 4. Порівняння алгоритмів за допомогою діаграми розмаху 
plt.boxplot(results, labels=names_models)
plt.title('Algorithm Comparison')
plt.ylabel('Accuracy Score')
plt.show()
#Крок 6-8 (Оцінка на тестових даних та прогноз)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# 1. Навчаємо модель SVM (або іншу найкращу) на тренувальних даних 
model_final = SVC(gamma='auto')
model_final.fit(X_train, Y_train)
# 2. Отримуємо прогноз на контрольній вибірці
predictions = model_final.predict(X_validation)
# 3. Виводимо показники якості [cite: 401-403]
print("Accuracy на контрольній вибірці:", accuracy_score(Y_validation, predictions))
print("\nМатриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("\nЗвіт про класифікацію:\n", classification_report(Y_validation, predictions))
# 4. КРОК 8: Прогноз для нової квітки 
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
prediction = model_final.predict(X_new)
print(f"\nПараметри нової квітки: {X_new}")
print(f"Спрогнозована мітка: {prediction}")
