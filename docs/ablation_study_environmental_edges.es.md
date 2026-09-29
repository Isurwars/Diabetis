# Estudio de Caso Académico y Guía de Defensa de Tesis: Topologías de Grafos Ambientales en Diabetes
## Estudio de Ablación: Homofilia Fenotípica Clínica ($k$-NN) frente a Co-habitación Comunitaria Geográfica (Intra-UPM)

---

### Resumen Ejecutivo para el Candidato a Licenciatura

En la investigación científica, un experimento que refuta una hipótesis intuitiva suele tener un valor metodológico superior al de mejoras marginales en métricas. Este documento formaliza el estudio de ablación realizado sobre la cohorte de la **ENSANUT 2018** en México ($N = 43,019$ ciudadanos adultos), diseñado para responder una interrogante central en ciencia de datos en salud:

> **Pregunta de Investigación**: *¿Incorporar aristas estructurales de co-habitación comunitaria geográfica (conectando a todos los habitantes de la misma Unidad Primaria de Muestreo / UPM) a un grafo de afinidad clínica ($k$-NN) mejora la discriminación diagnóstica de la Diabetes Tipo 2?*

* **Respuesta Empírica**: **No.** Agregar 458,866 aristas comunitarias incrementó el grafo de 571,866 a 1,042,853 aristas dirigidas (+82.4%), pero provocó un ligero decremento en las métricas de discriminación (ROC-AUC de prueba: `0.8356` $\rightarrow$ `0.8274`; PR-AUC de prueba: `0.3509` $\rightarrow$ `0.3319`).
* **Hallazgo Científico**: Para enfermedades metabólicas crónicas no transmisibles, **la homofilia fenotípica clínica domina con contundencia sobre la proximidad geográfica**. Los determinantes del hogar (tamaño de la familia, inseguridad alimentaria ELCSA) son altamente efectivos cuando se modelan como **atributos del nodo**, pero introducen ruido estructural cuando se expanden como **cliques espaciales densos**.

---

## 1. Formulación de Hipótesis

### La Intuición Epidemiológica
En epidemiología de enfermedades infecciosas (COVID-19, dengue, influenza), el contacto y la proximidad física son los vectores rectores del contagio. Para patologías crónicas como la Diabetes Mellitus Tipo 2, la literatura médica destaca con frecuencia el papel del "ambiente obesogénico":
* Hábitos alimentarios familiares compartidos y consumo de alimentos ultraprocesados.
* Calidad del agua de grifo y desiertos locales de acceso a frutas y verduras.
* Marginación comunitaria e infraestructura municipal deficiente.

Por lo tanto, la hipótesis intuitiva de la tesis postula:
$$\mathcal{H}_1: \text{Conectar a los co-habitantes de la misma comunidad } (UPM) \text{ capturará riesgos socioambientales compartidos, superando a los modelos basados únicamente en atributos clínicos.}$$

### La Hipótesis Nula (Principio de Parsimonia / Navaja de Ockham)
$$\mathcal{H}_0: \text{Las patologías crónicas no transmisibles dependen primariamente de la fisiopatología individual, edad y carga genética. Los cliques espaciales densos introducen heterofilia de etiquetas y sobre-suavizado.}$$

---

## 2. Metodología Experimental

Ambas configuraciones se entrenaron sobre la misma partición del 70% (30,142 ciudadanos) y se evaluaron sobre la misma partición de prueba del 15% por conglomerados espaciales no observados (6,584 ciudadanos en 938 conglomerados UPM):

```
Configuración A (Modelo Simplificado / Homofilia Fenotípica):
  • Aristas: 571,866 aristas dirigidas.
  • Topología: k-NN (k=10, Similitud Coseno sobre 24 atributos clínicos y del hogar).
  • Mecanismo: GATv2 puro sin incrustación de tipo de relación.

Configuración B (Modelo Multi-Relacional / Ablación Ambiental):
  • Aristas: 1,042,853 aristas dirigidas (583,987 k-NN clínico + 458,866 intra-UPM).
  • Topología: Grafo multi-relacional con incrustaciones de tipo de arista de 16 dimensiones.
  • Mecanismo: GATv2 Multi-Relacional.
```

---

## 3. Comparativa de Resultados Empíricos

| Métrica de Evaluación | Configuración A: $k$-NN Clínico | Configuración B: Multi-Relacional ($k$-NN + UPM) | Delta ($\Delta$) | Interpretación Académica |
| :--- | :--- | :--- | :--- | :--- |
| **Total de Aristas Dirigidas** | 571,866 | 1,042,853 | **+82.4%** | Fuerte incremento en memoria y costo de cómputo |
| **ROC-AUC en Prueba** | **`0.8356`** | `0.8274` | **-0.0082** | Ligera pérdida en capacidad de ordenamiento global |
| **PR-AUC en Prueba** | **`0.3509`** | `0.3319` | **-0.0190** | Menor precisión en niveles de exhaustividad clave |
| **Puntuación Brier (Calibración)** | **`0.1330`** | `0.1367` | **+0.0037** | Calibración probabilística ligeramente inferior |
| **Sensibilidad de Tamizaje** | 81.86% | **83.02%** | +1.16% | Ganancia marginal a expensas de más falsos positivos |
| **Especificidad de Tamizaje** | **71.45%** | 69.48% | -1.97% | Mayor tasa de falsas alarmas (1,799 vs 1,683) |
| **Exactitud Equilibrada ($F_1$-Max)** | **80.97%** | 80.57% | -0.40% | Sin beneficio en exactitud diagnóstica global |

---

## 4. Análisis Teórico: ¿Por Qué Fallaron las Aristas Ambientales?

¿Por qué agregar más de 450,000 conexiones geográficas reales perjudicó el desempeño? En teoría de grafos computacionales y aprendizaje automático, tres factores explican este resultado:

### 4.1 Heterofilia de Etiquetas en Conglomerados Espaciales
* Las Redes de Atención sobre Grafos destacan bajo **homofilia de etiquetas**, es decir, cuando los nodos conectados comparten la misma clase real ($y_u = y_v$).
* El grafo clínico $k$-NN maximiza la homofilia: un adulto de 58 años con hipertensión, sobrepeso y padres diabéticos se conecta con individuos de similar perfil biológico, quienes mayoritariamente son diabéticos o prediabéticos.
* En contraste, una Unidad Primaria de Muestreo ($UPM$) en México alberga en promedio a ~7 adultos de grupos etarios dispares: por ejemplo, un joven de 19 años sano y un adulto mayor de 82 años con diabetes crónica. Forzar una arista entre ellos conecta **nodos con etiquetas opuestas (heterofilia)**. Durante el paso de mensajes, las representaciones de los jóvenes diluyen el riesgo de los adultos mayores (**sobre-suavizado / over-smoothing**).

### 4.2 Dinámica de la Validación Espacial por Conglomerados
* El protocolo de evaluación implementó una **Partición por Conglomerados de UPM**: los 6,584 ciudadanos de prueba radican en localidades que **jamás fueron observadas durante el entrenamiento**.
* Durante el entrenamiento, la GNN puede aprender correlaciones locales porque observa las etiquetas de los vecinos del vecindario.
* Sin embargo, durante la prueba en una UPM no observada, **ninguno de los vecinos comunitarios tiene etiqueta conocida**. El paso de mensajes se limita a circular activaciones no supervisadas y ruidosas dentro de una pequeña isla aislada de 4 a 7 personas, amplificando la varianza local en lugar de generalizar el conocimiento epidemiológico.

### 4.3 Atributos de Nodo vs. Topología Estructural
* Las variables del hogar (tamaño de la familia, menores de edad, escala de inseguridad alimentaria ELCSA, apoyos alimentarios) **sí poseen valor predictivo**.
* No obstante, su efectividad se aprovecha al máximo cuando operan como **atributos del vector del nodo** ($\mathbf{x}_i \in \mathbb{R}^{24}$), permitiendo que las capas densas no lineales relacionen la vulnerabilidad doméstica con el riesgo metabólico.
* Transformar estos atributos en cliques densos de aristas saturó la topología sin aportar información independiente.

---

## 5. Preguntas Clave para la Defensa de Tesis y Respuestas Sugeridas

Durante el examen profesional de titulación ante el sínodo, estas son las preguntas típicas y cómo defenderlas:

### Pregunta 1: "¿Por qué utilizó una Red Neuronal sobre Grafos en lugar de conformarse con LightGBM o XGBoost, considerando que LightGBM obtuvo un ROC-AUC ligeramente superior (0.849 vs 0.835)?"
> **Defensa del Sustentante**:
> *"Si bien los árboles potenciados por gradiente logran métricas globales de ordenamiento levemente superiores en datos tabulares puros (0.849 vs 0.835), los modelos basados en árboles operan bajo el supuesto estricto de independencia entre individuos y son incapaces de ofrecer razonamiento basado en casos. En el contexto de tamizaje preventivo, GATv2 alcanzó una sensibilidad superior (83.0%–87.4% vs 77.1%) y ofrece de forma nativa explicabilidad clínica por similitud de pares: para cualquier paciente clasificado en alto riesgo, el médico puede inspeccionar la atención sobre casos similares en el registro de salud. Asimismo, un ensamble híbrido entre LightGBM y GATv2 unifica lo mejor de ambos mundos."*

### Pregunta 2: "¿Intentó conectar a los pacientes que viven en la misma comunidad o vivienda? ¿Cuáles fueron los resultados?"
> **Defensa del Sustentante**:
> *"Sí, ese análisis se formalizó en el Capítulo 5 como un estudio de ablación. Diseñamos un grafo multi-relacional con 458,866 aristas comunitarias intra-UPM adicionales. De manera contraintuitiva pero respaldada por la teoría de grafos, el modelo multi-relacional disminuyó levemente el ROC-AUC de 0.8356 a 0.8274 e incrementó el costo de cómputo en un 82%. Esto se explica por la alta heterofilia de etiquetas en los vecindarios (donde conviven jóvenes sanos con adultos mayores diabéticos), provocando sobre-suavizado en el paso de mensajes. Demostramos que las variables del entorno funcionan de manera óptima como atributos de entrada, mientras que la estructura topológica debe reservarse para la homofilia fenotípica."*

### Pregunta 3: "¿Cómo evitó la fuga de información espacial al evaluar encuestas complejas?"
> **Defensa del Sustentante**:
> *"La validación cruzada aleatoria tradicional es metodológicamente incorrecta en encuestas geoespaciales como la ENSANUT, ya que permite que individuos de prueba compartan localidad con individuos de entrenamiento, filtrando confusores ambientales locales. Implementamos una partición estricta por conglomerados sobre la variable UPM. El 100% de los 6,584 pacientes de prueba reside en 938 localidades jamás vistas durante el entrenamiento, garantizando que el ROC-AUC de 0.83+ reportado represente una generalización auténtica hacia nuevos municipios mexicanos."*

---

## 6. Comandos para Reproducir Ambos Escenarios

```bash
# 1. Ejecutar el modelo óptimo simplificado k-NN (Recomendado por defecto):
python scripts/run_training.py --graph-type knn --epochs 30 --device cpu

# 2. Ejecutar el experimento de ablación multi-relacional ambiental:
python scripts/run_training.py --graph-type multirelational --epochs 30 --device cpu

# 3. Comparar la inferencia de un paciente bajo ambas topologías:
python scripts/predict_patient.py --patient-idx 4 --graph-type knn
python scripts/predict_patient.py --patient-idx 4 --graph-type multirelational
```
