En la escuela Preuniversitario "Cuqui Bosch" en Santiago de Cuba (2023-2025), se observa que muchos estudiantes tienen 
dificultades en asignaturas como Matemáticas y Física, pero los profesores no pueden adaptar completamente 
su enseñanza a las necesidades individuales debido a la alta cantidad de alumnos por aula y la falta de recursos.


Un 30% de los estudiantes desaprobaron matemáticas en 2023 y la cosa no mejora. ¿Las causas? 
Los profes, con sus 30 alumnos por aula, no pueden dar atención personalizada; las evaluaciones 
llegan tarde cuando el problema ya está avanzado; y no hay forma de detectar a tiempo qué estudiantes necesitan ayuda específica.

Predecir desde primer trimestre quiénes van mal
Alertar a tiempo sobre los casos más críticos
Recomendar ejercicios específicos para cada estudiante

El resultado que esperamos:
Reducir la repitencia a menos del 15%, optimizar el trabajo de los profes y sacar 
el máximo provecho de los recursos existentes. Todo esto sin grandes inversiones, 
usando la tecnología de forma inteligente y adaptada a nuestra realidad escolar. 
¡Esto es resolver problemas con creatividad y pocos recursos, como solo los cubanos sabemos hacer!


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
estudiantes = [
    {"nombre": "Yanet Díaz", "matematicas": [38, 42, 45], "asistencia": 75},
    {"nombre": "Lázaro Mena", "matematicas": [28, 25, 30], "asistencia": 65},
    {"nombre": "Dayana Romero", "matematicas": [72, 80, 85], "asistencia": 90},
    {"nombre": "Omar García", "matematicas": [50, 48, 52], "asistencia": 80},
    {"nombre": "Elena Cabrera", "matematicas": [20, 22, 18], "asistencia": 60},
    {"nombre": "Carlos Martínez", "matematicas": [55, 60, 58], "asistencia": 85},
    {"nombre": "Ana Fernández", "matematicas": [65, 70, 68], "asistencia": 88},
    {"nombre": "José Rodríguez", "matematicas": [40, 35, 38], "asistencia": 70},
    {"nombre": "Laura Pérez", "matematicas": [75, 78, 80], "asistencia": 92},
    {"nombre": "Miguel Sánchez", "matematicas": [30, 28, 25], "asistencia": 62}
]
X = []
y = []
for estudiante in estudiantes:
    for nota in estudiante["matematicas"]:
        X.append([nota, estudiante["asistencia"]])
        y.append(0 if nota < 50 else 1)  
X = np.array(X)
y = np.array(y)
model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=0)
def predecir_rendimiento(nota, asistencia):
    prob = model.predict(np.array([[nota, asistencia]]), verbose=0)[0][0]
    riesgo = "Crítico" if prob < 0.3 else "Alto" if prob < 0.5 else "Moderado" if prob < 0.7 else "Bajo"
    return f"Riesgo: {riesgo} ({prob:.2%} de probabilidad de aprobar)"
print("Predicciones para estudiantes en riesgo:")
print(f"Lázaro Mena: {predecir_rendimiento(30, 65)}")
print(f"Elena Cabrera: {predecir_rendimiento(20, 60)}")
print(f"Miguel Sánchez: {predecir_rendimiento(25, 62)}")
print(f"\nPredicción para nuevo estudiante:")
print(predecir_rendimiento(45, 72))
