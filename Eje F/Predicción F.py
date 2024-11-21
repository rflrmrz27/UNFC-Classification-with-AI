import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo = joblib.load('modelo_arbol_decision.pkl')

# Función para obtener entradas válidas del usuario
def obtener_input_entero(prompt):
    while True:
        try:
            valor = int(input(prompt))
            if valor in [0, 1]:
                return valor
            else:
                print("Por favor, ingrese 1 para Sí o 0 para No.")
        except ValueError:
            print("Entrada no válida. Por favor, ingrese 1 para Sí o 0 para No.")

# Función para pedir datos del proyecto al usuario
def pedir_datos_proyecto():
    print("Ingrese los valores del proyecto (1 para Sí, 0 para No):")
    datos = {}
    
    # Lista de características requeridas
    caracteristicas = [
        'Producción Actual', 'Plan de desarrollo', 'Plan de desarrollo provisional',
        'Plan de desarrollo aprobado', 'Plan de desarrollo en proceso de aprobación',
        '¿Cuestión de tiempo para aprobar el plan de desarrollo?', '¿El Plan de desarrollo requiere más información?',
        '¿Es posible cumplir con los requerimientos?', 'Operaciones en progreso/Decisión final de inversión',
        'Solicitud de migración', 'Solicitud de migración aprobado', 'Migración en progreso',
        'Plan de evaluación', 'Plan de evaluación aprobado', 'Plan de evaluación en proceso de aprobación',
        'Se requiere más información para aprobar', 'Cumple los requerimientos para aprobar',
        'Plan de evaluación en progreso', 'Aprobado recientemente', '¿Es comercial?',
        'Tiene actividades pendientes', 'Informe de Evaluación y declaratoria de Comercialidad entregado',
        '¿Se pretende entregar el informe de Evaluación y declaratoria de Comercialidad?', 'Plan de Exploración',
        'Escenario Base', 'Escenario Incremental', 'Potencial convencional', 'Potencial No convencional'
    ]
    
    for caracteristica in caracteristicas:
        datos[caracteristica] = obtener_input_entero(f"{caracteristica}: ")
    
    # Crear un DataFrame con los datos ingresados
    datos_proyecto = pd.DataFrame([datos])
    return datos_proyecto

# Obtener los datos del usuario
nuevo_proyecto = pedir_datos_proyecto()

# Realizar la predicción y manejar posibles errores
try:
    prediccion = modelo.predict(nuevo_proyecto)[0]
    print(f"\nLa clasificación del proyecto es: {prediccion}")

    # Mostrar la explicación correspondiente a la predicción
    explicaciones = {
        'F1': "El proyecto es técnicamente viable y está listo para su desarrollo inmediato.",
        'F2': "El proyecto es técnicamente viable, pero hay cuestiones pendientes o condiciones que deben cumplirse antes de continuar.",
        'F3': "El proyecto no es técnicamente viable en este momento, pero puede haber potencial en el futuro."
    }

    print(explicaciones.get(prediccion, "Clasificación desconocida."))
except Exception as e:
    print("Ocurrió un error al realizar la predicción.")
    print(e)
