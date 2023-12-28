import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Función para cargar y preprocesar los datos
def cargar_preprocesar_datos(filepath):
    data = pd.read_csv(filepath)
    data_melted = data.melt(id_vars='estado', var_name='año', value_name='casos')
    umbral = data_melted['casos'].median()
    data_melted['clase'] = (data_melted['casos'] > umbral).astype(int)
    return data_melted

# Función para entrenar y evaluar un modelo
def entrenar_evaluar_modelo(modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    recall = recall_score(y_test, predicciones)
    return recall

# Cargar y preprocesar los datos
filepath = 'detecccion_diabetes_2000_2022.csv'  # Cambia esto a la ruta de tu archivo
data_preprocesada = cargar_preprocesar_datos(filepath)

# Preparando los datos para el modelado
X = data_preprocesada[['año']]  # Asumiendo el año como característica
y = data_preprocesada['clase']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Modelos a comparar
modelos = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Regresión Logística": LogisticRegression(),
    "Ridge Classifier": RidgeClassifier()
}

# Evaluando los modelos
resultados_recall = {nombre: entrenar_evaluar_modelo(modelo, X_train, y_train, X_test, y_test) for nombre, modelo in modelos.items()}

# Generando un gráfico de barras para comparar el recall de los modelos
plt.figure(figsize=(10, 6))
plt.bar(range(len(resultados_recall)), list(resultados_recall.values()), align='center')
plt.xticks(range(len(resultados_recall)), list(resultados_recall.keys()))
plt.xlabel('Modelo')
plt.ylabel('Recall')
plt.title('Comparación del Recall de Diferentes Modelos de Machine Learning')
plt.show()