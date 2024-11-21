import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#Cargar los datos 
data = pd.read_excel('ruta_al_archivo/Proyectos_EjeE.xlsx')
# Manejar valores faltantes
data = data.dropna()
#Preprocesar los datos
# Separar características (X) y la etiqueta (y)
X = data.drop(columns=['Clasificacion_E'])  # Todos los features menos la etiqueta
y = data['Clasificacion_E']  # La etiqueta que vamos a predecir
# Convertir variables categóricas en X (si las hay)
X = pd.get_dummies(X)
# Guardar las columnas después del preprocesamiento para usarlas en el script interactivo
columnas_entrenamiento = X.columns
columnas_entrenamiento.to_pickle('columnas_entrenamiento.pkl')
# Codificar las etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Guardar el LabelEncoder para usarlo en el script interactivo
joblib.dump(le, 'label_encoder.pkl')
# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Estandarizamos las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Guardar el scaler para usarlo en el script interactivo
joblib.dump(scaler, 'scaler.pkl')
# Obtener el número de clases
num_classes = len(np.unique(y_encoded))
#Definir el modelo de la red neuronal
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))  # Capa oculta con 64 neuronas
model.add(Dense(32, activation='relu'))  # Segunda capa oculta con 32 neuronas
model.add(Dense(num_classes, activation='softmax'))  # Capa de salida
# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, validation_split=0.1)
#Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Precisión en el conjunto de prueba: {test_acc}")
#Guardar el modelo
model.save('modelo_red_neuronal_eje_E.h5')

