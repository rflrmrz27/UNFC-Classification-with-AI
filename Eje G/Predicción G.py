import pandas as pd
from joblib import load  # Para cargar el modelo entrenado
import json  # Para cargar el orden de las características
# Carga del modelo entrenado
model = load('modelo_random_forest_eje_g.joblib')
# Carga del orden de las características utilizadas durante el entrenamiento
with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)
# Definición de las explicaciones de las categorías G
explicaciones_g = {
    1: "G1: Alta certeza geológica. La exploración, muestreo y pruebas geológicas son detalladas y confiables.",
    2: "G2: Moderada certeza geológica. Se han realizado estudios razonablemente confiables, pero con algunos supuestos.",
    3: "G3: Baja certeza geológica. La evidencia es limitada, no se puede verificar completamente la continuidad geológica.",
    4: "G4: Potencial basado en evidencia indirecta, con alto grado de incertidumbre."
}
# Función para obtener y validar la entrada del usuario
def obtener_valor_usuario(mensaje, opciones_validas):
    while True:
        try:
            valor = int(input(mensaje))
            if valor in opciones_validas:
                return valor
            else:
                print(f"Por favor, introduce uno de los siguientes valores: {opciones_validas}")
        except ValueError:
            print("Entrada no válida. Por favor, introduce un número entero.")
# Función principal para realizar la predicción
def realizar_prediccion_interactiva():
    print("Introduce los valores para cada una de las siguientes variables:\n")
    # Solicitud de valores al usuario con validación
    nivel_certeza_exploracion = obtener_valor_usuario(
        "Nivel de certeza en la exploración (0: Baja, 1: Moderada, 2: Alta): ", [0, 1, 2])
    evidencia_potencial = obtener_valor_usuario(
        "Evidencia potencial (0: No, 1: Sí): ", [0, 1])
    calidad_observacion_geologica = obtener_valor_usuario(
        "Calidad de la observación geológica (0: No confirmada, 1: Parcialmente confirmada, 2: Totalmente confirmada): ", [0, 1, 2])
    certidumbre_muestreo = obtener_valor_usuario(
        "Certidumbre del muestreo (0: Baja, 1: Moderada, 2: Alta): ", [0, 1, 2])
    distribucion_geologica_verificada = obtener_valor_usuario(
        "Distribución geológica verificada (0: No verificada, 1: Parcialmente verificada, 2: Totalmente verificada): ", [0, 1, 2])
    # Creación del DataFrame con los valores de entrada y el orden correcto de las columnas
    valores_entrada = pd.DataFrame([[
        nivel_certeza_exploracion,
        evidencia_potencial,
        calidad_observacion_geologica,
        certidumbre_muestreo,
        distribucion_geologica_verificada
    ]], columns=feature_columns)
    # Realización de la predicción con manejo de posibles excepciones
    try:
        prediccion = model.predict(valores_entrada)
        categoria_predicha = prediccion[0]
        # Mostrar el resultado de la predicción al usuario
        print(f"\nLa clasificación del proyecto en el Eje G es: G{categoria_predicha}")
        print(f"Explicación: {explicaciones_g[categoria_predicha]}")
    except Exception as e:
        print(f"Ocurrió un error durante la predicción: {e}")
# Ejecución de la función de predicción interactiva
if __name__ == "__main__":
    realizar_prediccion_interactiva()

