# programacion-avanzada-smile

Ejemplos en Java con la librería [SMILE](https://haifengl.github.io/) (*Statistical Machine
Intelligence and Learning Engine*): clasificación, regresión, clustering y PCA — material de la asignatura de Programación Avanzada del grado en Ingeniería Informática de la Universidad de Salamanca (USAL).

## Requisitos

- **JDK 25**
- **Maven**

SMILE 6.x usa características de API (records `Options`, `DataFrame` como clase) que requieren
Java 25. Si solo se dispone de un JDK 21 LTS, esta rama no compila; ver la guía para las
diferencias con SMILE 4.x.

## Ejecutar los ejemplos

Los ejemplos cargan los CSV de `data/` con rutas relativas, así que hay que lanzar Maven desde la
raíz del proyecto:

```bash
mvn compile
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo01CargaDatos
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo02Preprocesado
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo03Clasificacion
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo04Regresion
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo05Clustering
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo06PCA
```

## Contenido

| Ejemplo | Tema |
|---|---|
| [Ejemplo01CargaDatos](src/main/java/es/usal/smile/Ejemplo01CargaDatos.java) | Carga y exploración de datos con `DataFrame` |
| [Ejemplo02Preprocesado](src/main/java/es/usal/smile/Ejemplo02Preprocesado.java) | Estandarización, escalado, imputación y selección de características |
| [Ejemplo03Clasificacion](src/main/java/es/usal/smile/Ejemplo03Clasificacion.java) | KNN, LDA, regresión logística, random forest; métricas, validación cruzada, persistencia del modelo y encoders para variables categóricas |
| [Ejemplo04Regresion](src/main/java/es/usal/smile/Ejemplo04Regresion.java) | OLS y regresión regularizada/no lineal sobre datos sintéticos |
| [Ejemplo05Clustering](src/main/java/es/usal/smile/Ejemplo05Clustering.java) | K-means, DBSCAN, clustering jerárquico; índices de validación externa |
| [Ejemplo06PCA](src/main/java/es/usal/smile/Ejemplo06PCA.java) | Análisis de componentes principales y proyección |

Cada ejemplo es un `main()` independiente y autocontenido; [Utiles.java](src/main/java/es/usal/smile/Utiles.java)
reúne el único código compartido (partición entrenamiento/test y formato de salida por consola).

### Datos

- `data/iris.csv` — el conjunto clásico multiclase, usado en clasificación, clustering y PCA.
- `data/viviendas.csv` — datos sintéticos de vivienda generados con una relación lineal conocida
  más ruido gaussiano, para comprobar que OLS recupera los coeficientes reales.
- `data/viviendas_sucias.csv` — variante "sucia" del anterior, sin usar todavía en los ejemplos.
- `data/prestamos.csv` — datos sintéticos de aprobación de préstamos, con columnas numéricas y
  categóricas (`tipo_empleo`, `vivienda`, `historial_crediticio`); se usa para probar los
  *encoders* de variables categóricas (`LEVEL`, `ONE_HOT`) frente a modelos que aceptan
  categóricas sin codificar.

## Guía de SMILE

[src/GUIA-SMILE.md](src/GUIA-SMILE.md) es una guía práctica más extensa de la API de SMILE 6.3.0:
carga de datos, preprocesado, clasificación, regresión, clustering, PCA y errores frecuentes.

## Licencia

[GPLv3](LICENSE)
