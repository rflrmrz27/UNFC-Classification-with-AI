# Importamos las librerías necesarias
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Cargamos el modelo previamente entrenado
model = load_model('modelo_red_neuronal_eje_E.h5')

# Cargamos el scaler, el LabelEncoder y las columnas usadas durante el entrenamiento
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
columnas_entrenamiento = pd.read_pickle('columnas_entrenamiento.pkl')

# Definimos las explicaciones para cada categoría
explicaciones = {
    'E1.1': "El desarrollo es viable en base a las condiciones actuales y suposiciones realistas de las condiciones futuras.",
    'E1.2': "El desarrollo no es viable basado en las condiciones actuales del proyecto, pero se hace viable a través de subsidios gubernamentales y otras consideraciones.",
    'E2.1': "No todas las contingencias económicas, sociales y ambientales han sido resueltas, pero existe alta probabilidad de que se resuelvan en un futuro previsible.",
    'E2.2': "Los problemas aún no se han resuelto, pero existe una alta probabilidad de resolución con un esfuerzo activo para resolverlos.",
    'E2.3': "No se están tomando medidas activas, pero hay una probabilidad media de que los problemas se resuelvan en un futuro previsible.",
    'E3.1': "Existen estimaciones de proyectos futuros, pero no se utilizarán o consumarán operaciones.",
    'E3.2': "La viabilidad no se puede determinar debido a información insuficiente.",
    'E3.3': "No hay perspectivas razonables de que el proyecto se realice en el futuro previsible."
}

# Función para solicitar entradas de manera segura
def solicitar_entero(prompt, opciones_validas):
    while True:
        try:
            valor = int(input(prompt))
            if valor in opciones_validas:
                return valor
            else:
                print(f"Por favor, ingresa uno de los siguientes valores: {opciones_validas}")
        except ValueError:
            print("Entrada no válida. Por favor, ingresa un número entero.")

# Función para realizar la predicción interactiva
def realizar_prediccion_interactiva():
    print("Introduce los valores para cada una de las siguientes variables:")
    
    # Obtener valores del usuario
    area_restringida = solicitar_entero("¿El proyecto está localizado en un área restringida? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    flora_fauna = solicitar_entero("¿Flora y fauna enlistadas en la NOM-059-SEMARNAT-2010? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    ordenamiento_territorial = solicitar_entero("¿Hay Programa de Ordenamiento Ecológico Territorial? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    uso_critico_suelo = solicitar_entero("¿Hay uso crítico de suelo? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    presencia_indigena = solicitar_entero("¿Presencia de comunidades indígenas? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    region_indigena = solicitar_entero("¿Hay alguna región indígena? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    pertenencia_social = solicitar_entero("¿Hay pertenencia social y de la tierra? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    marginalizacion = solicitar_entero("¿Hay marginalización? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    interferencia_economica = solicitar_entero("¿El proyecto interfiere con alguna actividad económica? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    afectacion_agua = solicitar_entero("¿Hay afectación al proyecto por el agua? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    afectacion_legal = solicitar_entero("¿Hay alguna afectación por las variables legales? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    permisos_ambientales = solicitar_entero("¿Hay permisos y aprobaciones ambientales? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    evaluaciones_sociales = solicitar_entero("¿Hay evaluaciones sociales? (0: No, 1: Quizás, 2: Sí): ", [0, 1, 2])
    
    # Crear un DataFrame con los valores ingresados
    valores_entrada = {
        'area_restringida': area_restringida,
        'flora_fauna': flora_fauna,
        'ordenamiento_territorial': ordenamiento_territorial,
        'uso_critico_suelo': uso_critico_suelo,
        'presencia_indigena': presencia_indigena,
        'region_indigena': region_indigena,
        'pertenencia_social': pertenencia_social,
        'marginalizacion': marginalizacion,
        'interferencia_economica': interferencia_economica,
        'afectacion_agua': afectacion_agua,
        'afectacion_legal': afectacion_legal,
        'permisos_ambientales': permisos_ambientales,
        'evaluaciones_sociales': evaluaciones_sociales
    }
    
    valores_entrada_df = pd.DataFrame([valores_entrada]) 
    # Aplicar el mismo preprocesamiento que en el entrenamiento
    valores_entrada_df = pd.get_dummies(valores_entrada_df)
    # Alinear las columnas con las del entrenamiento
    valores_entrada_df = valores_entrada_df.reindex(columns=columnas_entrenamiento, fill_value=0)
    # Escalar los valores de entrada
    valores_entrada_escalados = scaler.transform(valores_entrada_df)
    # Realizar la predicción
    prediccion = model.predict(valores_entrada_escalados)
    categoria_predicha_index = prediccion.argmax()
    categoria_predicha = le.inverse_transform([categoria_predicha_index])[0]
    # Mostrar el resultado y la explicación
    print(f"\nLa predicción de clasificación para este proyecto en el eje E es: {categoria_predicha}")
    print(f"Explicación: {explicaciones.get(categoria_predicha, 'No se encontró una explicación para esta categoría.')}")
    
# Ejecutar la función interactiva
if __name__ == "__main__":
    realizar_prediccion_interactiva()
