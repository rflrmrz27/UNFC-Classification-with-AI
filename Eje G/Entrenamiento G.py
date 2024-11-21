import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump  # Para guardar el modelo entrenado
import json  # Para guardar el orden de las características
# Cargar de datos 
data = pd.read_excel('Proyectos_EjeG.xlsx')
# Verificación de valores nulos en el conjunto de datos
if data.isnull().sum().any():
    print("Advertencia: Hay valores nulos en el conjunto de datos. Por favor, revisar y limpiar los datos antes de continuar.")
    # Opcional: data = data.dropna()  # Eliminar filas con valores nulos
# Separarlas características (X) de la etiqueta (y)
X = data.drop(columns=['Clasificacion_G'])  # Variables de entrada
y = data['Clasificacion_G']  # Variable objetivo
# Verificación del equilibrio de las clases en la variable objetivo
print("Distribución de clases en y:\n", y.value_counts())
# Guardar el orden de las características para uso futuro
feature_columns = X.columns.tolist()
with open('feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)
# División de los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Definición del modelo Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
# Entrenamiento del modelo
model.fit(X_train, y_train)
# Realización de predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
# Evaluación del modelo
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
# Guardar el modelo entrenado para su uso en predicciones futuras
dump(model, 'modelo_random_forest_eje_g.joblib')
