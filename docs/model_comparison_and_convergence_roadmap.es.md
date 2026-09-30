# Comparativa de Modelos, Progresión de Benchmarks y Hoja de Ruta para Optimizar la Convergencia

**Proyecto**: Redes Neuronales de Grafos y Ensambles Tabulares para Tamizaje Epidemiológico de Diabetes en Adultos Mexicanos  
**Conjunto de Datos**: Encuesta Nacional de Salud y Nutrición (ENSANUT 2018) — Cohorte Adultos ($N = 43,019$)  
**Variable Objetivo**: Diagnóstico Médico Previo de Diabetes Mellitus ($P3\_1 = 1$, Prevalencia: $10.59\%$, $4,555$ casos positivos)  
**Protocolo de Evaluación**: Partición Espacial por Conglomerados de UPM ($N_{\text{entrenamiento}} = 30,142; N_{\text{validación}} = 6,293; N_{\text{prueba}} = 6,584$)

---

## 1. Resumen Ejecutivo y Trayectoria de Progresión

Este documento presenta una síntesis técnica y exhaustiva de todas las opciones arquitectónicas exploradas a lo largo de esta investigación de tesis, documenta la progresión cuantitativa de las métricas predictivas a través de cinco fases experimentales y define una hoja de ruta rigurosa para acelerar la velocidad de convergencia, suavizar el paisaje de pérdidas de optimización y maximizar la capacidad discriminativa del sistema.

```
       [Fase 1: Modelos Base]
       Modelos Tabulares Clásicos (24 variables crudas)
       LightGBM: ROC 0.8491 | PR 0.3781
                   │
                   ▼
       [Fase 2: GNN Inicial]
       GATv2 sobre Grafo Clínico k-NN (k=10)
       ROC 0.8263 | PR 0.3364 | Brier 0.1393
                   │
                   ▼
       [Fase 3: Ablación Topológica]
       Incorporación de 458k Aristas Comunitarias (UPM)
       ROC 0.8274 | PR 0.3319 (Sobre-suavizado por Heterofilia)
                   │
                   ▼
       [Fase 4: Ingeniería de Atributos Clínicos]
       43 Atributos Diseñados (Interacciones Metabólico-Edad y Dosis Genética)
       GATv2:    ROC 0.8403 (+0.0140) | PR 0.3600 (+0.0236) | Brier 0.1286
       LightGBM: ROC 0.8552 (+0.0061) | PR 0.3884 (+0.0103)
                   │
                   ▼
       [Fase 5: Ensambles Híbridos Tabular + Grafo]
       Mezcla Tri-Modelo y Meta-Aprendizaje Apilado (Stacking)
       ROC 0.8563 | PR 0.3938 | Brier 0.1372 | Sensibilidad 84.18%
```

---

## 2. Progresión Experimental: Tabla Maestra de Benchmarks

Todos los modelos en todas las fases fueron evaluados sobre la **misma partición de prueba por conglomerados de 6,584 ciudadanos** (689 diabéticos diagnosticados y 5,895 controles no diabéticos) residentes en Unidades Primarias de Muestreo (UPM) completamente inéditas:

### Tabla Cuantitativa de Progresión Histórica

| Fase | Modelo / Arquitectura | Espacio de Atributos | Topología de Entrada | ROC-AUC Prueba | PR-AUC Prueba | Puntuación Brier | Sensibilidad (Tamizaje) | Especificidad |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fase 1** | Regresión Logística | 24 Crudos | Tabular ($i.i.d.$) | `0.8401` | `0.3505` | `0.1482` | 74.30% | 76.80% |
| **Fase 1** | XGBoost (Ponderado) | 24 Crudos | Tabular ($i.i.d.$) | `0.8471` | `0.3773` | `0.1532` | 76.50% | 77.20% |
| **Fase 1** | LightGBM Base | 24 Crudos | Tabular ($i.i.d.$) | `0.8491` | `0.3781` | `0.1563` | 77.10% | 77.80% |
| **Fase 2** | GATv2 (Solo k-NN) | 24 Crudos | Grafo Clínico $k$-NN ($k=10$) | `0.8263` | `0.3364` | `0.1393` | 81.86% | 66.40% |
| **Fase 3** | GATv2 Multi-Relacional | 24 Crudos | Relacional ($k\text{-NN} + \text{UPM}$) | `0.8274` | `0.3319` | `0.1367` | 83.02% | 69.48% |
| **Fase 4** | Regresión Logística | 43 Diseñados | Tabular ($i.i.d.$) | `0.8465` | `0.3682` | `0.1420` | 76.80% | 77.10% |
| **Fase 4** | **GATv2 (k-NN)** | **43 Diseñados** | **Grafo Clínico $k$-NN ($k=10$)** | **`0.8403`** | **`0.3600`** | **`0.1286`** | **83.45%** | **70.18%** |
| **Fase 4** | XGBoost (Ponderado) | 43 Diseñados | Tabular ($i.i.d.$) | `0.8525` | `0.3902` | `0.1488` | 78.40% | 78.60% |
| **Fase 4** | LightGBM Diseñado | 43 Diseñados | Tabular ($i.i.d.$) | `0.8552` | `0.3884` | `0.1528` | 79.20% | 78.90% |
| **Fase 5** | **Ensamble 1: LGBM + GATv2** | 43 Diseñados | Mezcla de Probabilidades | `0.8558` | `0.3885` | `0.1431` | **84.91%** | 72.10% |
| **Fase 5** | **Ensamble 2: Tri-Modelo** | 43 Diseñados | Mezcla ($\text{LGB}+\text{XGB}+\text{GAT}$) | **`0.8560`** | **`0.3938`** | **`0.1372`** | **84.18%** | **74.60%** |
| **Fase 5** | **Ensamble 3: Meta-LR (Stacking)** | 43 Diseñados | Meta-Regresión Logística | **`0.8563`** | `0.3933` | `0.1638` | 82.58% | 76.20% |
| **Fase 5** | Ensamble 4: Fusión Latente | 43 Tab + 64 Lat | $[X \mathbin{\Vert} h_{\text{GNN}}] \rightarrow \text{LightGBM}$ | `0.8499` | `0.3869` | `0.1480` | 78.90% | 78.10% |

---

## 3. Análisis Crítico de Opciones Exploradas y Hallazgos Científicos

### 3.1 Opción 1: Árboles de Decisión Potenciados por Gradiente (LightGBM y XGBoost)
* **Mecanismo**: Partición recursiva y ortogonal sobre variables continuas y categóricas ponderadas por el factor de expansión de la encuesta (`scale_pos_weight = 8.44`).
* **Resultado Empírico**: Alta capacidad de discriminación en variables tabulares (`ROC-AUC 0.8552`, `PR-AUC 0.3884`).
* **Limitación**: Los árboles producen **probabilidades mal calibradas** con alta penalización Brier (`0.1528` a `0.1563`) y carecen de interpretabilidad basada en casos clínicos de referencia.

### 3.2 Opción 2: Red de Atención en Grafos (GATv2) sobre Grafo de Afinidad $k$-NN
* **Mecanismo**: GATv2 de dos capas con atención dinámica $\alpha_{ij}$ calculada sobre similitud coseno en el espacio fenotípico.
* **Resultado Empírico**: ROC-AUC competitivo (`0.8403`), excelente sensibilidad de tamizaje (**`83.45%`**) y el **menor error de calibración probabilística (Puntuación Brier `0.1286`)** entre todos los modelos aislados.
* **Valor Clínico**: Permite Razonamiento Basado en Casos (Case-Based Reasoning), extrayendo los 5 pacientes vecinos más influyentes para justificar el diagnóstico ante el médico.

### 3.3 Opción 3: Expansión Topológica mediante Aristas Comunitarias (UPM)
* **Mecanismo**: Inclusión de $458,866$ aristas espaciales conectando a todos los ciudadanos de la misma UPM geográfica.
* **Resultado Empírico**: **Degradación ligera en el rendimiento** (`ROC-AUC` cayó de `0.8356` $\rightarrow$ `0.8274`; `PR-AUC` cayó de `0.3509` $\rightarrow$ `0.3319`) con un costo computacional $82\%$ mayor.
* **Explicación Científica (Lección Teórica)**:
  1. *Heterofilia de Etiquetas*: Las enfermedades metabólicas crónicas no se transmiten por proximidad geográfica inmediata; en una misma manzana conviven jóvenes atléticos sanos de 20 años y ancianos diabéticos de 75 años. El paso de mensajes sobre clanes espaciales diluye los atributos fisiológicos.
  2. *Topología en Prueba*: En UPMs no observadas en entrenamiento, los nodos de prueba forman islas no etiquetadas sin supervisión directa que propagar.

### 3.4 Opción 4: Ingeniería Avanzada de Atributos Clínicos y Epidemiológicos
* **Mecanismo**: Se incorporaron 19 variables diseñadas basadas en curvas de deterioro metabólico y dosis genética acumulativa:
  - $\text{CargaGenética} = 2.0 \cdot \text{Padre} + 2.0 \cdot \text{Madre} + 1.5 \cdot \text{Hermano}$
  - $\text{Edad} \times \text{PesoHabitual}$, $\text{Edad} \times \text{Hipertensión}$, $\text{Edad} \times \text{Dislipidemia}$
  - $\text{ConteoComorbilidades} = \text{Obesidad} + \text{Hipertensión} + \text{Dislipidemia}$
  - $\text{VulnerabilidadSocioeconómica} = \text{CarenciaBienes} + \text{InseguridadAlim} + \text{AyudaAlim}$
* **Resultado Empírico**:
  - GATv2 experimentó un **crecimiento sustancial**: ROC-AUC aumentó $+0.0140$ (`0.8263` $\rightarrow$ `0.8403`) y PR-AUC aumentó $+0.0236$ (`0.3364` $\rightarrow$ `0.3600`).
  - La importancia de atributos coronó a $\text{Edad} \times \text{PesoHabitual}$ como la división más discriminante de toda la encuesta (644 divisiones).

### 3.5 Opción 5: Ensamblaje Híbrido Tabular + Grafo
* **Mecanismo**: Fusión ponderada de las probabilidades generadas por los árboles con la variedad regularizada y calibrada de GATv2.
* **Resultado Empírico**:
  - El Ensamble Tri-Modelo logró el mejor balance global: **`ROC-AUC 0.8560`**, **`PR-AUC 0.3938`**, reduciendo el error Brier a **`0.1372`** con una sensibilidad de **`84.18%`**.
  - El Meta-Aprendiz por Apilamiento (Regresión Logística sobre probabilidades) alcanzó la cima con **`ROC-AUC 0.8563`**.

---

## 4. Hoja de Ruta Técnica: Cómo Mejorar Aún Más la Convergencia y Optimización

Para acelerar el entrenamiento, estabilizar las curvas de pérdida y superar la barrera de `ROC-AUC 0.860`, se definen las siguientes estrategias concretas:

### 4.1 Dinámica de Optimización y Programación de Tasa de Aprendizaje

1. **Calentamiento Lineal con Decaimiento Coseno (OneCycleLR / Warmup)**:
   - *Problema*: En las primeras 3 épocas de GATv2, los pesos aleatorios de los mecanismos de atención $\mathbf{W}_{\text{att}}$ provocan gradientes desestabilizadores en las representaciones de los nodos.
   - *Solución*: Implementar 5 épocas de calentamiento donde la tasa de aprendizaje suba linealmente de $10^{-5}$ a $5 \times 10^{-3}$, seguido de decaimiento coseno hasta $10^{-6}$.
   - *Impacto Esperado*: Elimina la varianza de gradiente inicial y acelera la convergencia en ~35%.

2. **Recorte de Norma de Gradientes (Gradient Clipping)**:
   - *Solución*: Agregar `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` para evitar picos producidos por nodos centrales con alto grado de conectividad.

3. **Minimización Consciente de la Agudeza (SAM - Sharpness-Aware Minimization)**:
   - *Mecanismo*: SAM busca activamente mínimos anchos y planos en el paisaje de pérdida:
     $$\min_{\mathbf{w}} \max_{\|\mathbf{\epsilon}\|_2 \le \rho} \mathcal{L}(\mathbf{w} + \mathbf{\epsilon})$$
   - *Impacto Esperado*: Previene el sobreajuste a UPMs de entrenamiento y mejora la generalización en clusters de prueba en $+0.005$ ROC-AUC.

---

### 4.2 Regularización Estructural de Grafos y Suavizado del Paisaje de Pérdida

1. **DropEdge: Aumento de Datos Topológico (Rong et al., ICLR 2020)**:
   - *Formulación*: En cada época de entrenamiento, descartar aleatoriamente un porcentaje de aristas con probabilidad $p = 0.15$:
     $$\mathcal{E}_{\text{entrenamiento}}^{(t)} = \text{Bernoulli}(1 - p_{\text{drop}}) \odot \mathcal{E}$$
   - *Mecanismo*: Actúa como regularizador estructural de paso de mensajes, previniendo el colapso de representaciones hacia centroides idénticos.

2. **Conexiones Residuales de Entrada (Ego-Skip / Jumping Knowledge)**:
   - *Formulación*: Proyectar el vector original $x_i$ directamente hacia la capa clasificadora final:
     $$z_i = \text{LayerNorm}\left(h_i^{(2)} + \mathbf{W}_{\text{skip}} x_i\right)$$
   - *Mecanismo*: Asegura que los factores individuales determinantes (edad, historial familiar directo) no se diluyan ante vecinos con fenotipos limítrofes.

3. **Métrica Ponderada Clínicamente para el Grafo $k$-NN**:
   - *Mecanismo*: En lugar de similitud coseno con pesos unitarios para todas las variables, usar una matriz de ponderación diagonal $\mathbf{M} = \text{diag}(w_1, \dots, w_D)$ que asigne mayor peso a biomarcadores ($w_{\text{edad}} = 3.0, w_{\text{genética}} = 3.0, w_{\text{peso}} = 2.5$) y menor peso a bienes materiales ($w_{\text{bienes}} = 0.5$).

---

### 4.3 Formulación de Pérdidas de Ordenamiento y Desbalance

1. **Pérdida Subrogada de Ordenamiento AUC por Pares (Pairwise Surrogate AUC Loss)**:
   - *Formulación*: Optimizar de forma directa una aproximación cuadrática suave de la estadística Wilcoxon-Mann-Whitney sobre pares de diabéticos ($\mathcal{P}$) y controles ($\mathcal{N}$):
     $$\mathcal{L}_{\text{AUC}}(\mathbf{\theta}) \approx \frac{1}{|\mathcal{P}| |\mathcal{N}|} \sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}} \max\left(0, 1 - (f_\theta(x_i) - f_\theta(x_j))\right)^2$$
   - *Pérdida Compuesta*: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}} + \lambda_{\text{AUC}} \mathcal{L}_{\text{AUC}}$.

2. **Suavizado de Etiquetas (Label Smoothing, $\epsilon = 0.05$)**:
   - Convierte etiquetas duras $\{0, 1\}$ en probabilidades suaves $\{\epsilon, 1 - \epsilon\}$, impidiendo que la red sature logits en pacientes pre-diabéticos ambiguos.

---

### 4.4 Muestreo Estocástico por Mini-Lotes (PyG NeighborLoader)

1. **Migración a Mini-Lotes con NeighborLoader**:
   - *Estado Actual*: Ejecución por grafo completo en CPU ($43,019$ nodos, ~1.5 seg/época).
   - *Propuesta*: Carga por mini-lotes con `torch_geometric.loader.NeighborLoader` (tamaño de lote: 1,024 nodos, tamaños de subgrafo: $[15, 10]$).
   - *Beneficio*: El ruido estocástico del mini-lote ayuda a escapar puntos de ensilladura (saddle points) y permite entrenar en GPU sin limitaciones de memoria VRAM.

---

### 4.5 Optimización Bayesiana de Hiperparámetros con Optuna

Para exprimir las capacidades de LightGBM y XGBoost, un barrido bayesiano con 100 iteraciones sobre la partición de validación optimizará:
* `num_leaves` $\in [15, 63]$
* `max_depth` $\in [4, 8]$
* `learning_rate` $\in [0.01, 0.08]$
* `min_child_samples` $\in [20, 100]$
* `subsample` y `colsample_bytree` $\in [0.6, 0.9]$
* Regularización L1/L2 (`reg_alpha`, `reg_lambda`) $\in [10^{-3}, 10.0]$

---

## 5. Matriz Resumen de Impacto Técnico Esperado

| Dimensión de Mejora | Técnica Concreta | Cuello de Botella Mitigado | Complejidad de Implementación | Ganancia Esperada (ROC/PR) | Ganancia en Velocidad |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Optimización** | OneCycleLR + Calentamiento | Inestabilidad inicial de atención | Baja (10 líneas) | $+0.002$ ROC | **+35% más rápido** |
| **Optimización** | Gradient Clipping (máx 1.0) | Picos en nodos de alto grado | Mínima (2 líneas) | $+0.001$ ROC | Previene divergencia |
| **Regularización** | DropEdge ($p=0.15$) | Sobre-suavizado en épocas tardías | Baja | $+0.004$ ROC / $+0.010$ PR | Previene sobreajuste |
| **Arquitectura** | Conexión Residual Ego-Skip | Dilución de variables clave | Baja | $+0.003$ ROC | +15% más rápido |
| **Topología** | Grafo $k$-NN Ponderado | Ruido de distancia en bienes | Media | $+0.005$ ROC / $+0.012$ PR | Grafo más homofílico |
| **Función Pérdida** | Pérdida Subrogada AUC por Pares | Desajuste de entropía cruzada | Media | $+0.006$ ROC / $+0.018$ PR | Optimización de orden |
| **Muestreo** | PyG NeighborLoader | Límite de procesamiento CPU | Media | $+0.002$ ROC | **+400% más rápido (GPU)** |
| **Ensambles** | Búsqueda Bayesiana Optuna | Hiperparámetros subóptimos | Baja-Media | $+0.004$ ROC / $+0.008$ PR | N/A (Búsqueda offline) |
