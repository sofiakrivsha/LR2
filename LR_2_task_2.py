from sklearn.svm import SVC
kernels = [
    {'name': 'Поліноміальне (Poly)', 'kernel': 'poly', 'degree': 8}, 
    {'name': 'Гаусове (RBF)', 'kernel': 'rbf', 'degree': None},     
    {'name': 'Сигмоїдальне (Sigmoid)', 'kernel': 'sigmoid', 'degree': None} 
]
for k in kernels:
    if k['kernel'] == 'poly':
        clf = SVC(kernel=k['kernel'], degree=k['degree'], random_state=0, max_iter=2000)
    else:
        clf = SVC(kernel=k['kernel'], random_state=0, max_iter=2000)
    # Використовуємо OneVsOne як у завданні 
    model = OneVsOneClassifier(clf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 
    acc = accuracy_score(y_test, y_pred)
    print(f"Ядро: {k['name']} | Точність (Accuracy): {round(acc*100, 2)}%")
