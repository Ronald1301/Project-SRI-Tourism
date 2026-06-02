# 🎬 GUION FINAL – DEFENSA SISTEMA DE RECUPERACIÓN DE INFORMACIÓN (TURISMO CUBA)

---

# 🎤 🟢 INTRO (0:00 – 0:40)

🗣️ VOZ:

“En este video presento un sistema de recuperación de información enfocado en turismo en Cuba.

El sistema combina múltiples técnicas modernas como recuperación híbrida, embeddings, re-ranking, expansión de consultas, feedback del usuario y generación de respuestas mediante RAG.

Además, integra búsqueda web automática cuando la información local no es suficiente.”

🖥️ ACCIÓN:

* Mostrar frontend ya abierto
* Cursor listo en barra de búsqueda

---

# 🌍 1. DOMINIO + BLOQUEO (0:40 – 1:30)

🖥️ ESCRIBIR:

```
precio del dólar
```

🗣️ VOZ:

“El sistema primero aplica un detector de dominio.
Nuestro dominio es turismo en Cuba, por lo que consultas fuera de este contexto son bloqueadas.”

💡 SUBTÍTULO:

> Clasificación previa → evita procesamiento innecesario

---

# ⚠️ 2. CONSULTA DUDOSA (1:30 – 2:10)

🖥️ ESCRIBIR:

```
clima en varadero
```

🗣️ VOZ:

“Este es un caso ambiguo.
Aunque menciona un lugar de Cuba, no es claramente turismo.

El sistema puede marcarla como dudosa o manejarla con menor confianza.”

💡 SUBTÍTULO:

> Clasificador semántico → manejo de incertidumbre

---

# 🔎 3. RECUPERACIÓN LOCAL (2:10 – 3:30)

🖥️ ESCRIBIR:

```
hoteles en varadero
```

🗣️ VOZ:

“Aquí entra el pipeline principal.

El sistema realiza recuperación híbrida combinando:

* TF-IDF + LSI (modelo clásico)
* Embeddings semánticos
* Fusión mediante RRF”

💡 SUBTÍTULO:

> RRF = combinación robusta de rankings

🗣️ CONTINÚA:

“Esto permite encontrar resultados relevantes incluso cuando no hay coincidencias exactas.”

---

# 🧠 4. EXPANSIÓN + SEMÁNTICA (3:30 – 4:30)

🖥️ ESCRIBIR:

```
qué hacer en trinidad de noche
```

🗣️ VOZ:

“El sistema aplica expansión de consultas utilizando sinónimos, n-gramas y PRF.

Esto mejora el recall y permite capturar mejor la intención del usuario.”

💡 SUBTÍTULO:

> “vida nocturna”, “actividades”, etc.

---

# 👍 5. FEEDBACK (4:30 – 5:30)

🖥️ ACCIÓN:

* Dar LIKE a un resultado
* Repetir búsqueda

🗣️ VOZ:

“El sistema incorpora feedback explícito.

Un like incrementa el score del documento, mientras que un dislike lo elimina del ranking.”

💡 SUBTÍTULO:

> Aprendizaje basado en interacción

---

# 🌐 6. BÚSQUEDA WEB (CLAVE) (5:30 – 7:00)

🖥️ ESCRIBIR:

```
hoteles en varadero con tarifas actuales para junio de 2026
```

🗣️ VOZ:

“Esta consulta requiere información actualizada.

El sistema detecta insuficiencia en el corpus local mediante:

* bajo score
* pocos resultados
* baja confianza”

💡 SUBTÍTULO:

> Insufficiency Policy activada

🗣️ CONTINÚA:

“Entonces se activa automáticamente la búsqueda web.

Los resultados son filtrados, integrados al ranking y almacenados para uso futuro.”

---

# ⚙️ 7. INDEXACIÓN Y EMBEDDINGS (7:00 – 8:30)

🗣️ VOZ (SIN ESCRIBIR):

“El sistema utiliza dos tipos de índices:

* Índice invertido para TF-IDF/LSI
* Índice vectorial con FAISS

Los documentos se procesan con:

* limpieza
* tokenización
* stemming
* normalización”

🗣️ CONTINÚA:

“Para embeddings usamos Sentence Transformers, y la búsqueda se realiza con similitud coseno sobre FAISS HNSW, lo que permite alta eficiencia.”

---

# 🔄 8. FLUJO END-TO-END (8:30 – 9:30)

🖥️ HACER UNA BÚSQUEDA CUALQUIERA

🗣️ VOZ:

“El flujo completo es:

1. Usuario ingresa consulta
2. Detección de dominio
3. Preprocesamiento
4. Expansión
5. Recuperación híbrida
6. Fusión RRF
7. Re-ranking
8. Feedback
9. Detección de insuficiencia
10. Búsqueda web (si aplica)
11. Generación RAG
12. Visualización en frontend”

💡 SUBTÍTULO:

> Pipeline completo del sistema

---

# 🤖 9. RAG + CONTROL DE ALUCINACIONES (9:30 – 11:00)

🖥️ USAR UNA CONSULTA ANTERIOR

🗣️ VOZ:

“El módulo RAG toma los documentos más relevantes y los pasa como contexto al generador.

Para evitar alucinaciones:

* solo se usa contexto recuperado
* se limita el prompt
* se priorizan documentos por ranking”

💡 SUBTÍTULO:

> Generación basada en evidencia

---

# 🎯 10. RANKING (11:00 – 12:00)

🗣️ VOZ:

“El ranking final combina múltiples señales:

* similitud semántica
* coincidencia léxica
* feedback
* especificidad
* calidad del documento”

---

# 📊 11. EVALUACIÓN (12:00 – 12:40)

🗣️ VOZ:

“El sistema fue evaluado con métricas estándar:

* Precision
* Recall
* F1
* NDCG
* MRR

Mostrando mejoras frente a modelos básicos.”

---

# ⚠️ 12. DEFICIENCIAS (12:40 – 13:30)

🗣️ VOZ:

“Entre las limitaciones:

* dependencia del corpus
* costo computacional
* latencia en búsqueda web”

---

# 🚀 13. CIERRE (13:30 – 14:00)

🗣️ VOZ:

“En resumen, este sistema integra recuperación clásica, semántica, búsqueda web y generación de lenguaje en un solo pipeline.

No solo responde consultas, sino que mejora progresivamente su conocimiento.”

---

# 🧠 FRASE FINAL

“Este enfoque es similar al utilizado en motores de búsqueda modernos y sistemas avanzados de information retrieval.”

---

# 🎥 🧩 CONSEJOS PARA GRABAR (IMPORTANTE)

* Graba a **1080p**
* Cursor visible SIEMPRE
* Escribe lento (como si enseñaras)
* Pausas cortas después de cada búsqueda
* No leas → explica natural
* Sonríe en la voz (sí se nota)

---

# 🔥 PRO TIP (LO QUE TE HACE DESTACAR)

Cuando hagas la búsqueda web, di:

🗣️
“Esto es clave: el sistema no depende solo de su base local…
puede aprender en tiempo real.”

👉 Eso sube nivel de defensa MUCHO.

---
