# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib

# Cargar los datos desde el archivo Excel
file_path = 'ruta_a_tu_archivo/EjeF.xlsx'
eje_f_data = pd.read_excel(file_path)

# Separar las características (X) y la variable objetivo (y)
# Eliminamos columnas no relevantes para el modelo
X = eje_f_data.drop(columns=["Proyecto", "Clasificación Eje F"])
y = eje_f_data["Clasificación Eje F"]

# Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo de árbol de decisión con hiperparámetros ajustados
decision_tree_model = DecisionTreeClassifier(
    criterion='entropy',           # Usar entropía para calcular la ganancia de información
    max_depth=8,                   # Limitar la profundidad para evitar sobreajuste
    min_samples_leaf=5,            # Mínimo de muestras en una hoja
    random_state=42                # Semilla para reproducibilidad
)

# Entrenar el modelo con los datos de entrenamiento
decision_tree_model.fit(X_train, y_train)

# Evaluar el modelo con los datos de prueba
y_pred = decision_tree_model.predict(X_test)

# Generar y mostrar el reporte de clasificación
report = classification_report(y_test, y_pred)
print("Reporte de clasificación:\n", report)

# Guardar el modelo entrenado para usarlo futuro
joblib.dump(decision_tree_model, 'modelo_arbol_decision.pkl')
