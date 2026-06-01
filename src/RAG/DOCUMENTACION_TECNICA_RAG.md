# Documentación técnica del módulo RAG

## 1. Propósito del módulo

El módulo RAG (Retrieval-Augmented Generation) tiene como objetivo responder consultas en lenguaje natural utilizando evidencia recuperada desde el corpus local del sistema y, cuando esa evidencia no resulta suficiente, ampliar dinámicamente el contexto con información obtenida desde la web.

Su función central es reducir la dependencia de respuestas generativas sin soporte documental, favoreciendo salidas trazables, consistentes y ancladas en fuentes concretas.

## 2. Rol dentro del sistema

El módulo RAG actúa como capa de orquestación entre:

- el subsistema de recuperación de información,
- la base de datos vectorial,
- el repositorio documental local,
- el mecanismo de búsqueda web de apoyo,
- y el generador de lenguaje natural.

Desde el punto de vista funcional, el módulo recibe una consulta del usuario, localiza evidencia relevante, decide si esa evidencia es suficiente y, finalmente, sintetiza una respuesta en lenguaje natural.

## 3. Entrada y salida del módulo

### Entrada principal

- Consulta formulada por el usuario en lenguaje natural.
- Parámetros de control del proceso de recuperación, como la cantidad de evidencias a considerar.
- Corpus documental persistido localmente.
- Señales semánticas y léxicas derivadas del índice local.

### Salida principal

- Respuesta textual final.
- Evidencia recuperada y ordenada por relevancia.
- Contexto utilizado para generar la respuesta.

## 4. Fases del proceso RAG

El módulo opera en dos grandes fases:

### 4.1 Fase de recuperación local

En esta fase se consulta primero el corpus ya disponible en el sistema. La recuperación combina distintas señales de relevancia para localizar los fragmentos más útiles desde el punto de vista semántico y textual.

Esta etapa persigue tres objetivos:

- responder rápidamente cuando la información ya está en el corpus,
- minimizar la necesidad de consultar fuentes externas,
- y seleccionar evidencia suficiente para la generación.

### 4.2 Fase de ampliación dinámica

Cuando la evidencia local no alcanza para responder con calidad, el sistema activa una ampliación dinámica de información desde la web.

Esta fase añade documentos nuevos al flujo de recuperación, los valida, los normaliza y los incorpora al conjunto de evidencias disponibles para esa consulta.

El diseño busca que la respuesta no dependa exclusivamente de una base estática y que el sistema pueda cubrir consultas con información parcialmente ausente en el corpus local.

## 5. Estrategia de recuperación híbrida

La recuperación utilizada por el módulo es híbrida, es decir, combina dos familias de señales:

- una señal semántica basada en vectores densos,
- una señal clásica basada en representaciones léxicas y espacio latente.

### 5.1 Recuperación semántica

La recuperación semántica permite localizar documentos que expresan la misma intención de búsqueda con vocabulario distinto. Esto es importante en turismo, donde una misma necesidad puede aparecer formulada con sinónimos, nombres alternativos de lugares o descripciones indirectas.

### 5.2 Recuperación clásica

La recuperación clásica aporta robustez cuando la consulta contiene términos muy específicos, nombres propios, lugares concretos o expresiones con coincidencia lexical fuerte.

### 5.3 Fusión de resultados

Ambas señales se combinan mediante una estrategia de fusión de rankings para producir una lista final más estable. Este enfoque mejora:

- cobertura,
- resiliencia ante consultas ambiguas,
- y calidad del ordenamiento final.

## 6. Decisión de suficiencia documental

No toda recuperación local es suficiente para generar una respuesta útil. Por eso el módulo incorpora un criterio de suficiencia que evalúa si los documentos recuperados ofrecen evidencia adecuada.

La evaluación busca detectar casos como:

- resultados muy débiles o genéricos,
- falta de coincidencia temática,
- baja diversidad documental,
- o evidencia insuficiente para sostener una respuesta confiable.

Cuando la evidencia no alcanza un umbral razonable, el sistema activa la expansión web.

## 7. Ampliación web dinámica

La ampliación web se usa como mecanismo de apoyo, no como sustituto del corpus local.

Esta etapa tiene el propósito de:

- descubrir documentos adicionales relevantes,
- enriquecer el contexto disponible,
- y compensar vacíos del corpus inicial.

La información obtenida desde la web se normaliza antes de integrarse al proceso, de forma que se mantenga coherencia estructural con el resto de documentos del sistema.

## 8. Integración con la generación

Una vez seleccionado el conjunto de evidencias, el módulo construye un contexto compacto y lo entrega al generador de respuesta.

La generación está guiada por las evidencias recuperadas y no por conocimiento libre del modelo. Esto obliga a que la respuesta:

- se apoye en documentos concretos,
- cite o refleje fuentes recuperadas,
- y evite inferencias no respaldadas por el contexto.

## 9. Selección de fragmentos relevantes

Antes de generar la respuesta, el sistema no pasa todo el contenido recuperado sin filtrar. En su lugar, selecciona los fragmentos más útiles dentro de cada documento.

Esto permite:

- reducir ruido,
- evitar contexto excesivo,
- y centrar la generación en evidencias más informativas.

La selección favorece segmentos con relación directa con la consulta y penaliza fragmentos demasiado cortos, genéricos o poco informativos.

## 10. Construcción del prompt

La construcción del prompt sigue una política estricta de control de evidencia.

El prompt:

- establece que la respuesta debe ser fiel al contexto,
- prohíbe inventar información no respaldada,
- prioriza claridad y síntesis,
- y obliga a mantener el idioma de salida del sistema.

Esta estrategia mejora la trazabilidad de la respuesta y reduce la probabilidad de alucinaciones.

## 11. Persistencia y actualización del contexto

Cuando la búsqueda web aporta nuevos documentos relevantes, estos no solo se usan en memoria para una respuesta puntual. También se persisten e indexan para que puedan formar parte del corpus disponible en futuras consultas.

Esto convierte al módulo RAG en un sistema parcialmente adaptativo:

- aprende de las ampliaciones web,
- enriquece su base documental,
- y reduce la necesidad de repetir consultas externas si la información ya fue incorporada.

## 12. Justificación técnica del diseño

El diseño del módulo responde a necesidades concretas del dominio:

- el turismo requiere cobertura semántica amplia,
- las consultas de usuario suelen ser ambiguas o poco exactas,
- y la información útil puede estar distribuida entre fuentes locales y externas.

Por eso se eligió una arquitectura RAG con:

- recuperación híbrida,
- ampliación web bajo demanda,
- evaluación de suficiencia documental,
- y generación anclada en evidencia.

Este enfoque balancea:

- precisión,
- flexibilidad,
- cobertura,
- y control de calidad en la respuesta final.

## 13. Limitaciones actuales

Aun con esta arquitectura, existen limitaciones naturales:

- la calidad final depende de la calidad del corpus y de la web consultada,
- una mala consulta del usuario puede degradar la recuperación,
- la ampliación web añade latencia,
- y la fusión de rankings no garantiza por sí sola una respuesta perfecta.

Aun así, el diseño ofrece una base sólida para un sistema de recuperación y generación orientado a evidencia.

## 14. Resumen

El módulo RAG integra recuperación híbrida, expansión web dinámica y generación condicionada por evidencia para responder consultas turísticas con mayor solidez documental. Su diseño busca mantener un equilibrio entre rapidez, cobertura semántica, calidad de respuesta y capacidad de adaptación ante consultas para las que el corpus local no sea suficiente.
