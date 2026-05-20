# Frontend Roadmap - Sistema de Recuperación de Información (SRI)

## 1. Objetivo

Desarrollar una interfaz de usuario en Flet que permita realizar consultas en lenguaje natural y visualizar resultados provenientes de un sistema de recuperación de información.

El frontend actuará como una capa de orquestación, sin implementar lógica de recuperación, delegando esta responsabilidad al backend desarrollado en FastAPI.

---

## 2. Arquitectura

### 2.1 Enfoque

Se adopta una arquitectura desacoplada basada en cliente-servidor:

- Frontend: Flet (Python)
- Backend: FastAPI
- Comunicación: HTTP (JSON)

El frontend gestiona:
- Entrada del usuario
- Estado de la interfaz
- Renderizado de resultados

El backend gestiona:
- Procesamiento de consultas
- Recuperación de información
- Ranking y scoring

---

### 2.2 Flujo del sistema

1. El usuario introduce una consulta en lenguaje natural
2. El frontend envía la consulta al backend vía API REST
3. El backend procesa la consulta según el modo seleccionado:
   - Vectorial
   - LSI
   - RAG
4. El backend retorna una respuesta estructurada en JSON
5. El frontend renderiza los resultados de forma visual

---

## 3. Estructura del Proyecto

src/frontend_flet/

- app.py
- state.py
- enums.py
- theme.py

- api/
  - client.py

- views/
  - search_page.py

- components/
  - search_bar.py
  - result_card.py
  - rag_answer_card.py
  - status_banner.py

---

## 4. Gestión de Estado

### 4.1 Enfoque

Se implementa un estado centralizado mediante una clase AppState.

Se evita el uso de múltiples variables booleanas, reemplazándolas por un Enum que representa los estados de la interfaz.

---

### 4.2 UIState (Enum)

Estados definidos:

- IDLE: estado inicial
- LOADING: consulta en proceso
- SUCCESS: resultados obtenidos
- EMPTY: sin resultados
- ERROR: error en la consulta

---

### 4.3 AppState

El estado global incluye:

- query: consulta actual
- mode: modo seleccionado (vectorial, lsi, rag)
- top_k: número de resultados
- results: lista de resultados
- answer: respuesta generada (RAG)
- prompt: prompt utilizado (opcional)
- ui_state: estado actual (Enum)
- error_message: mensaje de error

---

## 5. Integración con Backend (FastAPI)

### 5.1 Request

El frontend enviará únicamente:

{
  "query": "string",
  "mode": "vectorial | lsi | rag",
  "top_k": 5
}

---

### 5.2 Response

El backend devuelve:

- results: lista de documentos recuperados
- score: relevancia
- explanation: (opcional)
- answer: (solo en RAG)
- prompt: (opcional)
- error: (si ocurre)

---

### 5.3 Cliente API

El módulo api/client.py se encarga de:

- Enviar requests HTTP
- Manejar errores de red
- Parsear respuestas JSON

---

## 6. Diseño de Interfaz

### 6.1 Filosofía

- Diseño minimalista
- Enfoque en contenido
- Alta legibilidad
- Interacción simple

---

### 6.2 Tema Visual

- Modo oscuro
- Inspiración en ChatGPT
- Fondo oscuro
- Tarjetas con contraste suave
- Tipografía clara

---

## 7. Componentes

### 7.1 SearchBar

Responsabilidades:

- Input de consulta
- Selector de modo
- Selector de top-k
- Botón de búsqueda

---

### 7.2 ResultCard

Muestra:

- ranking (#)
- título
- snippet
- score
- tipo de contenido
- fuente
- URL

---

### 7.3 RagAnswerCard

Muestra:

- respuesta generada
- fuentes utilizadas
- prompt (opcional)

---

### 7.4 StatusBanner

Maneja:

- estado de carga
- errores
- estado vacío

---

## 8. Flujo de Interacción

### Proceso de búsqueda

1. Usuario escribe consulta
2. Presiona botón "Buscar"
3. Estado cambia a LOADING
4. Se realiza llamada a la API
5. Se recibe respuesta
6. Estado cambia a:
   - SUCCESS
   - EMPTY
   - ERROR

---

## 9. Roadmap de Implementación

### UI-01: Setup inicial
- Crear estructura del proyecto
- Configurar entorno Flet
- Crear layout base

---

### UI-02: Controles de búsqueda
- Implementar SearchBar
- Manejar eventos

---

### UI-03: Integración API
- Implementar cliente HTTP
- Conectar con FastAPI

---

### UI-04: Renderizado de resultados
- Implementar ResultCard
- Renderizar lista dinámica

---

### UI-05: Soporte RAG
- Implementar RagAnswerCard
- Mostrar respuesta generada

---

### UI-06: Estados de la UI
- Implementar UIState
- Manejar estados globales

---

### UI-07: Tema oscuro
- Definir estilos globales
- Aplicar diseño consistente

---

### UI-08: Mejora de UX
- Scroll
- feedback visual
- destacar resultados relevantes

---

## 10. Decisiones de Diseño

- Uso de Flet por integración con Python
- Uso de FastAPI para desacoplamiento
- Uso de Enum para estado consistente
- Arquitectura modular para mantenibilidad
- Interfaz de una sola pantalla para simplicidad

---

## 11. Mejoras Futuras

- Paginación
- Historial de consultas
- Comparación de modelos
- Métricas de evaluación