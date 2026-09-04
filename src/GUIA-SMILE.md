# Guía práctica de SMILE para Java

SMILE (*Statistical Machine Intelligence and Learning Engine*) es la librería de
aprendizaje automático más completa del ecosistema JVM. Cubre clasificación,
regresión, clustering, reducción de dimensión, NLP, álgebra lineal y, en los
módulos opcionales, deep learning e inferencia con LLMs. Es software libre bajo
licencia GPLv3.

Esta guía acompaña a los seis ejemplos del directorio `src/main/java/es/usal/smile`
y está escrita contra la **versión 6.3.0**, verificada sobre el código fuente de
la librería.

---

## 1. Versiones y requisitos

| Rama de SMILE | Java mínimo | Notas |
|---|---|---|
| 6.x (actual) | **Java 25** | API con records `Options`, `DataFrame` como clase |
| 4.x | Java 21 | Muy parecida a la 6.x |
| 3.x | Java 8 | Hiperparámetros vía `Properties` |
| 2.x | Java 8 | `smile.projection.PCA`, `DataFrame` como interfaz |

Si tus máquinas de laboratorio tienen un JDK 21 LTS, la rama 6.x **no compila**:
en ese caso hay que bajar a `4.x` y revisar las diferencias puntuales de la API.
Todo el código de estos ejemplos asume 6.3.0.

## 2. Instalación

Una sola dependencia basta para todo lo que se ve aquí: `smile-core` arrastra
transitivamente `smile-base`, donde viven `DataFrame`, la E/S y las matemáticas.

```xml
<dependency>
    <groupId>com.github.haifengl</groupId>
    <artifactId>smile-core</artifactId>
    <version>6.3.0</version>
</dependency>
```

En Gradle (Kotlin DSL):

```kotlin
implementation("com.github.haifengl:smile-core:6.3.0")
```

Módulos opcionales: `smile-plot` (gráficos Swing y Vega-Lite), `smile-nlp`,
`smile-deep` (LibTorch, requiere librerías nativas), `smile-kotlin`,
`smile-scala`.

SMILE usa SLF4J para las trazas; sin un *binding* en el classpath verás un aviso
al arrancar. El `pom.xml` de este proyecto incluye `slf4j-simple`.

Algunos algoritmos (aprendizaje de variedades, procesos gaussianos, MLP) van
bastante más rápido con BLAS/LAPACK nativos (`libopenblas-dev` y `libarpack2` en
Debian/Ubuntu), pero **no son necesarios**: hay implementación en Java puro.

### Ejecutar los ejemplos

```bash
mvn compile
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo01CargaDatos
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo03Clasificacion
```

Las rutas de los CSV son relativas al directorio del proyecto, así que ejecuta
Maven desde la raíz.

SMILE también se integra con **JShell**: descargando el paquete de release,
`./smile shell` abre una sesión con todas las clases preimportadas, muy cómoda
para demostraciones en clase.

---

## 3. Carga de datos

Todo gira alrededor de `smile.data.DataFrame`: una estructura **columnar y
tipada**. Cada columna tiene un `DataType` (`DoubleType`, `IntType`,
`StringType`...) y opcionalmente una `Measure` que indica que la variable es
categórica (`NominalScale`, `OrdinalScale`).

```java
DataFrame iris = Read.csv("data/iris.csv", "header=true");
```

> **Trampa número uno**: por defecto `CSVFormat.DEFAULT` **no** lee la cabecera y
> las columnas pasan a llamarse `V1`, `V2`, `V3`... Hay que pedirlo de forma
> explícita. La cadena de formato admite pares `clave=valor` separados por comas:
> `header=true`, `delimiter=\t`, `quote="`, `escape=\`, `comment=#`.

Otros lectores: `Read.arff`, `Read.json`, `Read.parquet`, `Read.arrow`,
`Read.avro`, `Read.sas`, `Read.libsvm`. `Read.data(ruta)` infiere el formato por
la extensión. Para escribir: `Write.csv`, `Write.arff`, `Write.arrow`,
`Write.object` (serialización de modelos).

Operaciones habituales:

```java
iris.nrow(); iris.ncol();          // dimensiones
iris.schema().fields();            // esquema: nombre, tipo, medida
iris.describe();                   // estadísticos por columna
iris.column("sepal_length").toDoubleArray();
iris.select("petal_length", "species");
iris.drop("species");
iris.filter("sepal_length", ">", "7.0");
iris.slice(0, 10);
iris.sample(20);
iris.dropna();  iris.fillna(0.0);
iris.toArray();                    // -> double[][]
```

### De texto a variable nominal: `factorize()`

Al leer un CSV, una columna de clases como `species` llega como texto. Los
algoritmos supervisados necesitan enteros con una escala nominal asociada:

```java
DataFrame iris = Read.csv("data/iris.csv", "header=true").factorize("species");
```

`factorize()` sin argumentos convierte **todas** las columnas de texto. Para
recuperar los nombres de las clases después:

```java
NominalScale escala = (NominalScale) iris.schema().field("species").measure();
String nombre = escala.level(prediccion);
```

Los ficheros ARFF ya traen la información de tipos, así que no hace falta
factorizar: es una molestia específica del CSV.

### La API de fórmulas

Herencia de R, y muy cómoda en clase:

```java
Formula formula = Formula.lhs("species");   // species ~ .  (todo lo demás)
Formula f2 = Formula.of("precio", "superficie", "antiguedad");

double[][] x = formula.x(datos).toArray();
int[] y = formula.y(datos).toIntArray();
double[] yNum = formula.y(datos).toDoubleArray();
```

---

## 4. Preprocesado

Todas las transformaciones siguen el patrón `fit` / `apply`, equivalente al
`fit`/`transform` de scikit-learn. **Se ajustan solo con el conjunto de
entrenamiento y se aplican después a entrenamiento y test**; ajustarlas con
todos los datos es fuga de información y sesga la evaluación.

| Clase | Qué hace | Cuándo |
|---|---|---|
| `Standardizer` | media 0, desviación 1 | por defecto para SVM, KNN, redes, PCA |
| `RobustStandardizer` | mediana e IQR | cuando hay valores atípicos |
| `Scaler` | escala a `[0, 1]` | rangos acotados |
| `WinsorScaler` | `[0,1]` recortando percentiles | atípicos extremos |
| `MaxAbsScaler` | divide por el máximo absoluto | datos dispersos (no rompe la esparsidad) |
| `Normalizer` | norma unidad **por fila** | texto, clustering de documentos |

```java
InvertibleColumnTransform std = Standardizer.fit(train);   // todas las numéricas
DataFrame trainStd = std.apply(train);
DataFrame testStd  = std.apply(test);
DataFrame original = std.invert(trainStd);                 // deshacer
```

`Normalizer` es la excepción: no aprende nada, se instancia directamente
(`new Normalizer(Normalizer.Norm.L2, columnas...)`).

**Imputación** (`smile.feature.imputation`): `SimpleImputer` (media/moda),
`KNNImputer`, `KMedoidsImputer`, `SVDImputer`.

**Codificación de categóricas**: `toArray(bias, encoder, columnas...)` con
`CategoricalEncoder.LEVEL` (índice del nivel, válido para árboles), `DUMMY`
(k−1 binarias, para modelos lineales) o `ONE_HOT` (k binarias).

**Encadenar transformaciones**:

```java
Transform pipeline = Transform.fit(train,
        d -> SimpleImputer.fit(d),
        d -> Standardizer.fit(d, "superficie", "antiguedad"));
DataFrame listo = pipeline.apply(train);
```

**Selección de características** (`smile.feature.selection`):
`SumSquaresRatio` (multiclase), `SignalNoiseRatio` (binaria),
`InformationValue`, `FRegression`, `GAFE` (algoritmo genético). Además, los
modelos de ensemble exponen `importance()` y `shap()`.

---

## 5. Clasificación

Hay dos estilos de API que conviven:

**Con fórmula y `DataFrame`** — modelos basados en árboles, que aceptan
variables categóricas sin codificar:

```java
RandomForest bosque = RandomForest.fit(formula, train, new RandomForest.Options(200));
int clase = bosque.predict(test.get(0));            // predict(Tuple)
```

**Con arrays** — KNN, SVM, LDA/QDA, regresión logística, MLP:

```java
KNN<double[]> knn = KNN.fit(xTrain, yTrain, 5);
int[] pred = knn.predict(xTest);
```

### Hiperparámetros: los records `Options`

Es el cambio de estilo más visible respecto a versiones antiguas. Cada
algoritmo declara su propio record anidado, con constructores abreviados:

```java
new RandomForest.Options(200);                       // solo nº de árboles
new RandomForest.Options(200, 3, 20, 100, 5);        // ntrees, mtry, maxDepth, maxNodes, nodeSize
new LogisticRegression.Options(0.1, 1E-5, 500);      // lambda, tolerancia, iteraciones
new GradientTreeBoost.Options(300);
```

### Métricas y validación

```java
Accuracy.of(yTest, pred);
ConfusionMatrix.of(yTest, pred);
```

En `smile.validation.metric` también hay `Precision`, `Recall`, `FScore`, `AUC`,
`LogLoss`, `MatthewsCorrelation`, `Sensitivity`, `Specificity`.

Los modelos de ensemble se autoevalúan con las muestras *out-of-bag*:
`bosque.metrics()` devuelve un `ClassificationMetrics` sin necesidad de test.

Validación cruzada:

```java
var resultado = CrossValidation.classification(10, formula, datos,
        (f, d) -> RandomForest.fit(f, d, opciones));
resultado.avg().accuracy();
resultado.std().accuracy();
```

Variantes: `CrossValidation.stratify(...)` mantiene la proporción de clases,
`CrossValidation.classification(5, 10, ...)` hace 5 repeticiones de 10-fold,
y existen `LOOCV` y `Bootstrap`.

### Persistencia

```java
Write.object(bosque, Path.of("modelo.sml"));
RandomForest recuperado = (RandomForest) Read.object(Path.of("modelo.sml"));
```

`Read.object` aplica una lista blanca de clases al deserializar, así que evita
los problemas de seguridad habituales de la serialización de Java.

---

## 6. Regresión

Misma estructura. `OLS.fit(formula, datos)` devuelve un `LinearModel` cuyo
`toString()` imprime un resumen al estilo de `summary()` en R: coeficientes,
error estándar, estadístico *t*, *p*-valores, R², R² ajustado y test F.

```java
LinearModel ols = OLS.fit(formula, train);
ols.intercept();  ols.coefficients();  ols.ttest();
ols.RSquared();   ols.adjustedRSquared();  ols.ftest();  ols.pvalue();
double[] pred = ols.predict(test);          // sobre un DataFrame completo
```

Regularizados: `RidgeRegression`, `LASSO`, `ElasticNet`, todos devolviendo
también un `LinearModel`. No lineales: `RegressionTree`, `RandomForest`,
`GradientTreeBoost`, `SVM`, `GaussianProcessRegression`, `RBFNetwork`, `MLP`,
`GAM`, `GLM`.

Métricas: `RMSE.of`, `MSE.of`, `MAD.of`, `R2.of`, `RSS.of`. Validación cruzada
con `CrossValidation.regression(...)`, cuyo `avg()` da un `RegressionMetrics`
con `rmse()`, `mad()`, `r2()`.

El dataset `viviendas.csv` es sintético, generado con una relación lineal
conocida más ruido gaussiano. Sirve para dos cosas en clase: comprobar que los
coeficientes estimados recuperan los verdaderos, y ver que un random forest
**pierde** frente a OLS cuando el modelo generador es lineal.

---

## 7. Clustering

Los algoritmos de partición devuelven un `CentroidClustering` (un record) o una
subclase de `Partitioning`. En ambos casos las etiquetas están en `group()`, y
los puntos de ruido se marcan con `Clustering.OUTLIER`.

```java
CentroidClustering<double[], double[]> km = KMeans.fit(x, 3, 100);
int[] etiquetas = km.group();
km.k();  km.distortion();  km.size(i);  km.center(i);  km.radius(i);
km.predict(nuevaMuestra);
```

- **Elegir k a mano**: método del codo sobre `distortion()`.
- **Elegir k automáticamente**: `XMeans.fit(x, kmax, maxIter)` (criterio BIC) o
  `GMeans.fit(...)` (test de gaussianidad).
- **Por densidad**: `DBSCAN.fit(x, minPts, radio)`, que además detecta ruido.
  También `HDBSCAN`, `DENCLUE`.
- **Jerárquico**: se construye el criterio de enlace y se corta el dendrograma.

  ```java
  var hc = HierarchicalClustering.fit(WardLinkage.of(x));
  int[] particion = hc.partition(3);      // por número de clusters
  int[] otra = hc.partition(2.5);         // por altura
  ```

- **Otros**: `KMedoids`, `KModes` (categóricas), `SpectralClustering`, `MEC`,
  `SIB`, `DeterministicAnnealing`.

Para comparar la partición obtenida con unas etiquetas verdaderas se usan
índices externos, invariantes a cómo se numeren los clusters:
`RandIndex`, `AdjustedRandIndex`, `NormalizedMutualInformation`,
`AdjustedMutualInformation`.

**Recuerda estandarizar antes** si las variables tienen escalas distintas: todos
estos métodos se basan en distancias.

---

## 8. PCA

Está en `smile.feature.extraction.PCA` (en la 2.x estaba en `smile.projection`;
es el error de importación más frecuente al seguir tutoriales antiguos).

```java
PCA pca = PCA.fit(x);        // matriz de covarianzas
PCA pcaCor = PCA.cor(x);     // matriz de correlaciones
```

Usa `cor()` cuando las variables estén en unidades distintas (equivale a
estandarizar antes); `fit()` cuando compartan unidad, porque estandarizar
destruye información sobre las varianzas relativas.

```java
Vector varianza   = pca.variance();                      // autovalores
Vector proporcion = pca.varianceProportion();
Vector acumulada  = pca.cumulativeVarianceProportion();
System.out.println(pca.loadings());                      // autovectores
```

Estos vectores son `smile.tensor.Vector`, se recorren con `.get(i)` y `.size()`.
La tabla de varianza explicada es el equivalente numérico del *scree plot*; con
`smile-plot` se dibuja con `new ScreePlot(pca.varianceProportion()).canvas().window()`.

Proyección:

```java
PCA p2 = pca.getProjection(2);        // 2 componentes
double[][] x2 = p2.apply(x);

PCA p95 = pca.getProjection(0.95);    // las que expliquen el 95% de la varianza
```

Variantes: `ProbabilisticPCA` (modelo de variables latentes, tolera ausentes),
`KernelPCA` (no lineal, O(n²) en memoria), `RandomProjection`, `GHA` (PCA
incremental). Para reducción no lineal seria, `smile.manifold` trae `TSNE`,
`UMAP`, `IsoMap`, `LLE` y `LaplacianEigenmap`.

---

## 9. Errores frecuentes

1. **Columnas llamadas `V1, V2...`** — falta `header=true` en `Read.csv`.
2. **`ClassCastException` o `toIntArray()` que falla en la clase** — falta
   `factorize()` sobre la columna de etiquetas leída de un CSV.
3. **`PCA` no encontrado** — está en `smile.feature.extraction`, no en
   `smile.projection` (2.x) ni en `smile.projection.PCA` de tutoriales viejos.
4. **Transformaciones ajustadas con todo el dataset** — fuga de información:
   ajusta con `train` y aplica a `test`.
5. **Resultados que cambian en cada ejecución** — fija `MathEx.setSeed(42)`.
6. **Clustering sin estandarizar** con variables de escalas dispares.
7. **`Options` frente a `Properties`** — en 6.x los hiperparámetros son records;
   los ejemplos de la web que pasan un `Properties` corresponden a la 3.x.
8. **La documentación oficial mezcla versiones**: varias páginas de
   `haifengl.github.io` siguen mostrando código de la 2.x. Ante la duda, el
   Javadoc de la versión concreta manda.

---

## 10. Ideas de uso en docencia

- **Programación Avanzada**: los ejemplos 3 a 6 cubren el bloque de aprendizaje
  automático clásico sin salir de Java, así que los alumnos no tienen que
  cambiar de lenguaje a mitad de asignatura. El ejemplo 4 sirve para discutir
  sesgo-varianza con datos de generador conocido.
- **Programación III**: el diseño de SMILE es un buen caso de estudio de POO
  moderna en Java — interfaces con métodos estáticos y por defecto
  (`Standardizer`, `Transform`), records para valor inmutable (`Options`,
  `CentroidClustering`, `ClassificationMetrics`), genéricos acotados en
  `Classifier<T>`, y expresiones lambda como estrategia de entrenamiento en la
  validación cruzada.
- Una práctica que funciona bien: dar el pipeline montado y pedir que sustituyan
  el clasificador y comparen con validación cruzada, discutiendo la desviación
  típica entre pliegues y no solo la media.

---

## 11. Referencias

- Web oficial: <https://haifengl.github.io/>
- Repositorio: <https://github.com/haifengl/smile>
- Javadoc por versión: <https://javadoc.io/doc/com.github.haifengl/smile-core>
- Guías por tema: `classification.html`, `regression.html`, `clustering.html`,
  `feature.html`, `validation.html` en la web oficial.
