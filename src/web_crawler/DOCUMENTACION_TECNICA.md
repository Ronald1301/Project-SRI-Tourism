# Documentación técnica del módulo `web_crawler`

## 1. Propósito del módulo

El módulo `src/web_crawler` implementa la capa de adquisición de documentos web del sistema de recuperación de información orientado al dominio turístico. Su objetivo no es solo descargar páginas, sino construir y mantener una base documental utilizable por los componentes de indexación, recuperación y generación de respuestas.

El diseño del módulo responde a dos necesidades complementarias:

- crear un corpus local inicial de documentos de alta calidad;
- ampliar ese corpus cuando la información local no alcanza para responder con suficiente confianza a una consulta.

Por ello, el módulo se organiza en dos fases operativas claramente diferenciadas.

---

## 2. Fase inicial: adquisición de datos y construcción del corpus

La primera fase tiene como objetivo reunir un corpus inicial estable, acotado y reutilizable. Esta fase corresponde al proceso de crawling principal y se ejecuta sobre un conjunto de sitios turístico-temáticos previamente definidos.

### 2.1. Objetivos funcionales

- recorrer sitios semilla y enlaces internos relevantes;
- respetar restricciones de dominio, esquema y patrones URL;
- obedecer `robots.txt`;
- evitar duplicados y re-descargas innecesarias;
- extraer texto utilizable a nivel documental;
- persistir el corpus en un formato incremental y reproducible;
- registrar el conjunto de URLs ya visitadas para no reprocesar información.

### 2.2. Criterios de calidad del corpus inicial

No toda página visitada se considera útil. El módulo incorpora filtros para conservar solo contenido con valor informativo:

- idioma compatible con la aplicación;
- longitud mínima suficiente para evitar fragmentos triviales;
- exclusión de páginas de navegación, formularios, archivos estáticos o contenido no textual;
- reducción de ruido generado por menús, pie de página, publicidad o bloques repetitivos.

### 2.3. Persistencia del corpus

El resultado de esta fase se almacena como documentos estructurados en `data/raw/documents.jsonl`. Cada línea representa un documento independiente, lo que facilita:

- lectura incremental;
- reanudación de ejecuciones posteriores;
- integración con procesos de preprocesamiento e indexación;
- trazabilidad del origen de cada documento.

Adicionalmente, el crawler mantiene un archivo de control de URLs visitadas en `src/web_crawler/visited_urls.txt`. Este archivo actúa como mecanismo de persistencia ligera para evitar redescargas y repetir trabajo ya realizado.

### 2.4. Concurrencia y seguridad de escritura

El crawling inicial puede ejecutarse de forma concurrente por sitio. Dado que distintos hilos pueden acceder a recursos compartidos, el módulo sincroniza la escritura en archivos comunes para evitar corrupción de datos y condiciones de carrera.

En particular:

- `documents.jsonl` se escribe en modo incremental;
- `visited_urls.txt` se actualiza con sincronización por ruta;
- las políticas de acceso impiden que varios crawlers escriban simultáneamente sobre el mismo recurso sin control.

---

## 3. Fase final: búsqueda web de apoyo

La segunda fase se activa cuando la información recuperada del corpus local no es suficiente para responder una consulta con calidad. Su finalidad no es reemplazar el repositorio local, sino ampliarlo temporalmente con evidencia reciente o ausente en el corpus interno.

### 3.1. Propósito operativo

Esta fase actúa como mecanismo de expansión de evidencia. El sistema:

- detecta que el corpus local no aporta contexto suficiente;
- ejecuta búsqueda web orientada a turismo y a Cuba;
- filtra los resultados antes de incorporarlos;
- extrae contenido útil de las páginas seleccionadas;
- si procede, agrega esos documentos al flujo de recuperación y generación.

### 3.2. Estrategia de búsqueda

La búsqueda web se apoya en DuckDuckGo como motor de descubrimiento inicial, pero no se queda en el listado de resultados: también sigue enlaces relevantes con una profundidad controlada.

El proceso busca equilibrar cobertura y precisión:

- la consulta se enriquece con contexto turístico;
- solo se consideran URLs que parezcan relevantes para el dominio temático;
- se limita la profundidad de exploración para mantener el proceso acotado;
- se aplica un tope máximo de páginas para evitar exploraciones largas o costosas.

### 3.3. Reglas de filtrado de la búsqueda web

Antes de usar un documento web como evidencia, el módulo aplica varias capas de control:

- importancia de URL respecto a la consulta;
- respeto de `robots.txt`;
- timeouts por solicitud para evitar bloqueos o saturación;
- mínima longitud textual;
- idioma en español o inglés;
- exclusión de páginas con exceso de boilerplate;
- exclusión de páginas con densidad excesiva de enlaces;
- filtro semántico basado en embeddings para comparar el contenido con:
  - un referente temático general;
  - la consulta concreta del usuario.

Estas reglas reducen el ruido y mejoran la probabilidad de que la información incorporada sea realmente útil.

---

## 4. Arquitectura lógica del módulo

El módulo se puede entender como una cadena de responsabilidades separadas:

1. configuración y políticas;
2. normalización y selección de URLs;
3. descarga de HTML;
4. extracción de contenido estructurado;
5. control de calidad documental;
6. persistencia incremental;
7. búsqueda web de apoyo;
8. integración con el sistema de recuperación superior.

Este diseño permite que el crawler no dependa de un único tipo de sitio, sino de reglas reutilizables y extensibles.

### 4.1. Presets por sitio

Cada sitio cuenta con su propia definición de:

- URL semilla;
- dominios permitidos;
- patrones de inclusión y exclusión;
- restricciones particulares del sitio.

Esto permite adaptar el crawling a sitios con estructuras heterogéneas sin alterar el motor general.

### 4.2. Políticas de acceso

Las políticas controlan:

- esquema permitido (`http` / `https`);
- dominio válido;
- patrones URL aceptados o descartados;
- respeto de `robots.txt`;
- normalización de enlaces relativos y absolutización;
- limpieza de fragmentos y parámetros inútiles.

### 4.3. Extracción documental

La extracción transforma HTML en un objeto documental estructurado con campos útiles para recuperación:

- identificador del documento;
- URL original;
- dominio;
- título;
- resumen o extracción breve;
- texto principal;
- conteo de palabras;
- idioma;
- metadatos auxiliares cuando existan.

Esto permite tratar cada página como una unidad documental recuperable.

---

## 5. Persistencia y trazabilidad

El módulo mantiene dos ideas centrales de persistencia:

### 5.1. Persistencia del corpus

Los documentos nuevos se anexan al corpus en formato JSONL. Este formato es conveniente porque:

- permite agregar documentos sin reescribir todo el archivo;
- conserva una línea por documento;
- simplifica el preprocesamiento posterior;
- facilita auditoría y depuración.

### 5.2. Persistencia de URLs visitadas

Las URLs visitadas se registran en un archivo plano compartido. Esta decisión reduce redescargas y mejora la eficiencia del crawling, especialmente en ejecuciones sucesivas.

---

## 6. Estrategia de calidad documental

El módulo no se limita a extraer texto. También intenta aproximarse a la idea de "contenido útil" dentro del dominio turístico.

Los criterios principales son:

- longitud suficiente del contenido;
- idioma útil para el sistema;
- baja proporción de elementos de plantilla;
- baja densidad de enlaces;
- relevancia temática respecto a la consulta o al sitio objetivo;
- compatibilidad semántica con el dominio del proyecto.

Esta estrategia busca minimizar ruido documental y maximizar la probabilidad de que el corpus sirva para recuperación semántica y respuesta contextual.

---

## 7. Integración con el resto del sistema

`web_crawler` no opera como un componente aislado. Sus salidas alimentan directamente:

- la base de datos vectorial;
- los índices de recuperación;
- el pipeline de RAG;
- el mecanismo de búsqueda web de apoyo.

En consecuencia, la calidad del crawler afecta de forma directa la calidad del sistema completo. Un corpus pobre degrada la recuperación, mientras que un corpus controlado y bien filtrado mejora la precisión de las respuestas.

---

## 8. Justificación técnica del diseño

La arquitectura escogida responde a criterios prácticos:

- **Modularidad**: separar configuración, políticas, extracción y persistencia simplifica mantenimiento.
- **Escalabilidad**: agregar nuevos sitios requiere solo nuevas definiciones de sitio, no reescribir el motor.
- **Robustez**: el respeto de `robots.txt`, los timeouts y los filtros de URL reducen fallos y comportamiento agresivo.
- **Eficiencia**: el control de páginas, profundidad y concurrencia evita exploraciones innecesarias.
- **Calidad semántica**: los filtros de idioma, longitud, boilerplate y similitud reducen el ruido documental.
- **Reutilización**: el corpus generado sirve tanto para indexación local como para ampliación web bajo demanda.

---

## 9. Resumen conceptual del flujo completo

### 9.1. Fase inicial

Se recolecta un corpus local inicial desde sitios turístico-temáticos, aplicando políticas de filtrado y persistencia.

### 9.2. Fase final

Si una consulta no puede resolverse con suficiente evidencia local, el sistema activa la búsqueda web de apoyo. Esa búsqueda amplía el universo documental de forma acotada, filtra los resultados y devuelve evidencia adicional que puede ser indexada y utilizada por el sistema de respuesta.

---

## 10. Conclusión técnica

El módulo `web_crawler` implementa una solución de adquisición documental orientada a dos objetivos: construir un corpus local inicial de calidad y, cuando sea necesario, expandirlo mediante búsqueda web controlada. La separación entre ambas fases permite combinar estabilidad, trazabilidad y extensibilidad, manteniendo al mismo tiempo un control estricto sobre ruido, duplicados, saturación de sitios y relevancia temática.