from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
# Список моделей 
models_income = []
models_income.append(('LR', LogisticRegression(solver='liblinear')))
models_income.append(('LDA', LinearDiscriminantAnalysis()))
models_income.append(('KNN', KNeighborsClassifier()))
models_income.append(('CART', DecisionTreeClassifier()))
models_income.append(('NB', GaussianNB()))
models_income.append(('SVM', SVC(gamma='auto')))
# Оцінка моделей
results_income = []
names_income = []
for name, model in models_income:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    # Використовуємо підмножину даних (наприклад, 10000) для прискорення обчислень
    cv_results = cross_val_score(model, X_final[:10000], y_final[:10000], cv=kfold, scoring='accuracy')
    results_income.append(cv_results)
    names_income.append(name)
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')
# Візуалізація порівняння [cite: 381-382]
import matplotlib.pyplot as plt
plt.boxplot(results_income, labels=names_income)
plt.title('Порівняння алгоритмів для Income Dataset')
plt.show()
