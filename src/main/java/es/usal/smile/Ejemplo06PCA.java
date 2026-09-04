package es.usal.smile;

import java.util.Arrays;

import smile.classification.KNN;
import smile.data.DataFrame;
import smile.feature.extraction.KernelPCA;
import smile.feature.extraction.PCA;
import smile.feature.extraction.ProbabilisticPCA;
import smile.io.Read;
import smile.manifold.KPCA;
import smile.math.MathEx;
import smile.math.kernel.GaussianKernel;
import smile.tensor.Vector;
import smile.validation.metric.Accuracy;

/**
 * Ejemplo 6: analisis de componentes principales.
 *
 * <p>En SMILE 3.x en adelante el PCA vive en {@code smile.feature.extraction}
 * (en la 2.x estaba en {@code smile.projection}). {@code PCA} hereda de
 * {@code Projection}, asi que una vez elegido el numero de componentes con
 * {@code getProjection} se proyecta con {@code apply}.</p>
 */
public class Ejemplo06PCA {

    public static void main(String[] args) throws Exception {

        MathEx.setSeed(42);

        DataFrame iris = Read.csv("data/iris.csv", "header=true").factorize("species");
        double[][] x = iris.drop("species").toArray();
        int[] y = iris.column("species").toIntArray();

        Utiles.separador("1. Ajuste del PCA");

        // PCA.fit  -> matriz de covarianzas
        // PCA.cor  -> matriz de correlaciones (equivale a estandarizar antes).
        //
        // Regla practica: si las variables estan en unidades distintas o con
        // escalas muy dispares, usar cor(); si comparten unidad, fit().
        PCA pca = PCA.fit(x);

        System.out.println("Media de cada variable (centro): " + pca.center());

        Utiles.separador("2. Varianza explicada");

        Vector varianza = pca.variance();                        // autovalores
        Vector proporcion = pca.varianceProportion();            // proporcion
        Vector acumulada = pca.cumulativeVarianceProportion();   // acumulada

        System.out.printf("%4s %14s %14s %14s%n", "PC", "VARIANZA", "PROPORCION", "ACUMULADA");
        for (int i = 0; i < proporcion.size(); i++) {
            System.out.printf("%4d %14.4f %13.2f%% %13.2f%%%n",
                    i + 1, varianza.get(i), 100 * proporcion.get(i), 100 * acumulada.get(i));
        }

        // Esta tabla es el equivalente numerico del grafico de sedimentacion
        // (scree plot). Con smile-plot se dibujaria con:
        //   new ScreePlot(pca.varianceProportion()).canvas().window();

        Utiles.separador("3. Cargas (loadings)");

        // Los autovectores: cuanto pesa cada variable original en cada
        // componente. Es lo que permite interpretar las componentes.
        System.out.println(pca.loadings());

        Utiles.separador("4. Proyeccion a 2 componentes");

        PCA proyeccion2D = pca.getProjection(2);
        double[][] x2 = proyeccion2D.apply(x);

        System.out.printf("Dimension original: %d -> proyectada: %d%n", x[0].length, x2[0].length);
        for (int i = 0; i < 5; i++) {
            System.out.printf("  muestra %d (clase %d): %s%n", i, y[i], Arrays.toString(x2[i]));
        }

        // Tambien se puede pedir por varianza explicada en lugar de por numero
        // de componentes: getProjection(0.95) devuelve las componentes
        // necesarias para retener el 95% de la varianza.
        PCA proyeccion95 = pca.getProjection(0.95);
        System.out.printf("Componentes para explicar el 95%% de la varianza: %d%n",
                proyeccion95.apply(x)[0].length);

        Utiles.separador("5. PCA sobre la matriz de correlaciones");

        PCA pcaCor = PCA.cor(x);
        Vector proporcionCor = pcaCor.varianceProportion();
        System.out.printf("Primera componente: %.2f%% (covarianzas) vs %.2f%% (correlaciones)%n",
                100 * proporcion.get(0), 100 * proporcionCor.get(0));

        Utiles.separador("6. PCA como paso previo a un clasificador");

        // Reducir dimension antes de clasificar: menos ruido, menos coste, y
        // en problemas con muchas variables suele mejorar la generalizacion
        // (efecto Hughes / maldicion de la dimensionalidad).
        Utiles.ParticionArrays particion = Utiles.split(x, y, 0.7);

        // IMPORTANTE: el PCA se ajusta SOLO con entrenamiento y se aplica
        // despues al test. Ajustarlo con todos los datos es fuga de informacion.
        PCA pcaTrain = PCA.fit(particion.xTrain()).getProjection(2);
        double[][] xTrain2 = pcaTrain.apply(particion.xTrain());
        double[][] xTest2 = pcaTrain.apply(particion.xTest());

        KNN<double[]> knnCompleto = KNN.fit(particion.xTrain(), particion.yTrain(), 5);
        KNN<double[]> knnReducido = KNN.fit(xTrain2, particion.yTrain(), 5);

        System.out.printf("KNN con 4 variables    accuracy = %.2f%%%n",
                100.0 * Accuracy.of(particion.yTest(), knnCompleto.predict(particion.xTest())));
        System.out.printf("KNN con 2 componentes  accuracy = %.2f%%%n",
                100.0 * Accuracy.of(particion.yTest(), knnReducido.predict(xTest2)));

        Utiles.separador("7. Variantes no lineales y probabilisticas");

        // PCA probabilistico: modelo de variables latentes, admite datos
        // ausentes y da una interpretacion generativa.
        ProbabilisticPCA ppca = ProbabilisticPCA.fit(x, 2);
        System.out.println("PPCA, primera muestra proyectada: " + Arrays.toString(ppca.apply(x)[0]));

        // Kernel PCA: aplica el truco del kernel para capturar estructura no
        // lineal. El coste es O(n^2) en memoria por la matriz de Gram.
        // Ojo: esta variante solo acepta DataFrame, no double[][].
        KernelPCA kpca = KernelPCA.fit(iris.drop("species"),
                new GaussianKernel(4.0), new KPCA.Options(2));
        System.out.println("KPCA, primera muestra proyectada: " + Arrays.toString(kpca.apply(x)[0]));

        // Otras tecnicas de reduccion de dimension en SMILE:
        //   smile.feature.extraction.RandomProjection  (Johnson-Lindenstrauss)
        //   smile.feature.extraction.GHA              (PCA incremental, Hebb)
        //   smile.manifold.{TSNE, UMAP, IsoMap, LLE, LaplacianEigenmap}
        //   smile.mds.{MDS, IsotonicMDS, SammonMapping}
    }
}
