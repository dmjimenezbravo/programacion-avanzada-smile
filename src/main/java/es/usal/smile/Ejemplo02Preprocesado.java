package es.usal.smile;

import java.util.Arrays;

import smile.data.CategoricalEncoder;
import smile.data.DataFrame;
import smile.data.transform.InvertibleColumnTransform;
import smile.data.transform.Transform;
import smile.feature.imputation.SimpleImputer;
import smile.feature.selection.SumSquaresRatio;
import smile.feature.transform.MaxAbsScaler;
import smile.feature.transform.Normalizer;
import smile.feature.transform.RobustStandardizer;
import smile.feature.transform.Scaler;
import smile.feature.transform.Standardizer;
import smile.feature.transform.WinsorScaler;
import smile.io.Read;

/**
 * Ejemplo 2: preprocesado de datos.
 *
 * <p>Todas las transformaciones siguen el mismo patron: un metodo estatico
 * {@code fit(DataFrame, columnas...)} aprende los parametros del conjunto de
 * entrenamiento y devuelve un {@code Transform} que se aplica despues con
 * {@code apply(DataFrame)}. Es el equivalente a fit/transform de sklearn, y la
 * regla es la misma: <b>ajustar solo con train y aplicar a train y a test</b>.</p>
 */
public class Ejemplo02Preprocesado {

    public static void main(String[] args) throws Exception {

        DataFrame iris = Read.csv("data/iris.csv", "header=true").factorize("species");

        Utiles.separador("1. Estandarizacion (media 0, desviacion 1)");

        // Si no se indican columnas, transforma todas las numericas.
        // Devuelve InvertibleColumnTransform: se puede deshacer con invert().
        InvertibleColumnTransform estandarizador = Standardizer.fit(iris,
                "sepal_length", "sepal_width", "petal_length", "petal_width");

        System.out.println(estandarizador);   // muestra la formula de cada columna
        DataFrame irisEstandarizado = estandarizador.apply(iris);
        System.out.println(irisEstandarizado.head(5));

        // Deshacer la transformacion
        System.out.println(estandarizador.invert(irisEstandarizado).head(3));

        Utiles.separador("2. Otros escalados disponibles");

        // Scaler: lleva cada variable al intervalo [0, 1]
        DataFrame escalado = Scaler.fit(iris, "sepal_length").apply(iris);
        System.out.println("Scaler [0,1]        -> " + escalado.head(3));

        // WinsorScaler: como Scaler pero recortando percentiles extremos.
        // Recomendable cuando hay valores atipicos.
        DataFrame winsor = WinsorScaler.fit(iris, 0.05, 0.95, "sepal_length").apply(iris);
        System.out.println("WinsorScaler 5%-95% -> " + winsor.head(3));

        // RobustStandardizer: resta la mediana y divide por el IQR.
        DataFrame robusto = RobustStandardizer.fit(iris, "sepal_length").apply(iris);
        System.out.println("RobustStandardizer  -> " + robusto.head(3));

        // MaxAbsScaler: divide por el maximo absoluto, no desplaza el origen
        // (util con datos dispersos, no destruye la esparsidad).
        DataFrame maxAbs = MaxAbsScaler.fit(iris, "sepal_length").apply(iris);
        System.out.println("MaxAbsScaler        -> " + maxAbs.head(3));

        // Normalizer: normaliza cada FILA a norma unidad (no cada columna).
        // No necesita fit porque no aprende nada de los datos.
        Normalizer normalizador = new Normalizer(Normalizer.Norm.L2,
                "sepal_length", "sepal_width", "petal_length", "petal_width");
        System.out.println("Normalizer L2       -> " + normalizador.apply(iris).head(3));

        Utiles.separador("3. Imputacion de valores ausentes");

        DataFrame sucias = Read.csv("data/viviendas_sucias.csv", "header=true");
        System.out.println("Antes de imputar:");
        System.out.println(sucias.head(8));

        // SimpleImputer: media para variables continuas, moda para categoricas.
        SimpleImputer imputador = SimpleImputer.fit(sucias);
        DataFrame imputadas = imputador.apply(sucias);
        System.out.println("Despues de imputar:");
        System.out.println(imputadas.head(8));

        // Alternativas mas sofisticadas en smile.feature.imputation:
        //   KNNImputer.fit(datos, k)        imputa con los k vecinos mas proximos
        //   KMedoidsImputer.fit(datos, k)   imputa con el medoide del cluster
        //   SVDImputer.fit(datos, k)        factorizacion de bajo rango

        Utiles.separador("4. Codificacion de variables categoricas");

        DataFrame conZona = imputadas.factorize("zona");

        // toArray(bias, encoder, columnas...) controla como se codifican
        // las columnas nominales al pasar a matriz numerica:
        //   LEVEL    -> el indice del nivel (0, 1, 2...). Valido para arboles.
        //   DUMMY    -> k-1 columnas binarias. Para modelos lineales (evita
        //               colinealidad con el termino independiente).
        //   ONE_HOT  -> k columnas binarias.
        double[][] nivel  = conZona.toArray(false, CategoricalEncoder.LEVEL,   "zona");
        double[][] dummy  = conZona.toArray(false, CategoricalEncoder.DUMMY,   "zona");
        double[][] oneHot = conZona.toArray(false, CategoricalEncoder.ONE_HOT, "zona");

        System.out.println("LEVEL   fila 0: " + Arrays.toString(nivel[0]));
        System.out.println("DUMMY   fila 0: " + Arrays.toString(dummy[0]));
        System.out.println("ONE_HOT fila 0: " + Arrays.toString(oneHot[0]));

        // El primer parametro (bias) añade una columna de unos, util para
        // implementar a mano modelos lineales con termino independiente.

        Utiles.separador("5. Encadenar transformaciones (pipeline)");

        // Transform.fit aplica los entrenadores uno detras de otro, ajustando
        // cada uno sobre la salida del anterior.
        Transform pipeline = Transform.fit(sucias,
                datos -> SimpleImputer.fit(datos),
                datos -> Standardizer.fit(datos, "superficie", "antiguedad", "distancia_centro"));

        System.out.println(pipeline.apply(sucias).head(5));

        Utiles.separador("6. Seleccion de caracteristicas");

        // Razon de sumas de cuadrados entre grupos / dentro de grupos.
        // Cuanto mayor, mejor separa esa variable las clases.
        SumSquaresRatio[] ratios = SumSquaresRatio.fit(iris, "species");
        for (SumSquaresRatio r : ratios) {
            System.out.printf("%-16s %8.3f%n", r.feature(), r.ratio());
        }

        // Otras opciones en smile.feature.selection:
        //   SignalNoiseRatio  -> problemas binarios
        //   InformationValue  -> valor de informacion (scoring crediticio)
        //   FRegression       -> test F para regresion
        //   GAFE              -> seleccion de subconjuntos con algoritmo genetico
        // Y los modelos de ensemble exponen importance() (ver Ejemplo03).
    }
}
