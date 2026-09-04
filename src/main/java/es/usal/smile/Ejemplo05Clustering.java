package es.usal.smile;

import java.util.Arrays;

import smile.clustering.CentroidClustering;
import smile.clustering.Clustering;
import smile.clustering.DBSCAN;
import smile.clustering.HierarchicalClustering;
import smile.clustering.KMeans;
import smile.clustering.XMeans;
import smile.clustering.linkage.WardLinkage;
import smile.data.DataFrame;
import smile.io.Read;
import smile.math.MathEx;
import smile.validation.metric.AdjustedRandIndex;
import smile.validation.metric.NormalizedMutualInformation;
import smile.validation.metric.RandIndex;

/**
 * Ejemplo 5: clustering (aprendizaje no supervisado).
 *
 * <p>Los algoritmos de particion devuelven un {@code CentroidClustering} (un
 * record) o una subclase de {@code Partitioning}. En ambos casos las etiquetas
 * asignadas estan en {@code group()}. Los puntos considerados ruido se marcan
 * con {@link Clustering#OUTLIER}.</p>
 *
 * <p>Usamos iris quitando la clase: sabemos que hay 3 especies, asi que
 * podemos comparar la particion obtenida con la verdadera usando indices
 * externos (Rand ajustado, informacion mutua normalizada).</p>
 */
public class Ejemplo05Clustering {

    public static void main(String[] args) throws Exception {

        MathEx.setSeed(42);

        DataFrame iris = Read.csv("data/iris.csv", "header=true").factorize("species");
        double[][] x = iris.drop("species").toArray();
        int[] especieReal = iris.column("species").toIntArray();

        Utiles.separador("1. K-medias");

        // fit(datos, k, maxIteraciones)
        CentroidClustering<double[], double[]> kmeans = KMeans.fit(x, 3, 100);
        System.out.println(kmeans);

        int[] grupos = kmeans.group();
        System.out.printf("Numero de clusters: %d%n", kmeans.k());
        System.out.printf("Distorsion (suma de distancias al centroide): %.3f%n", kmeans.distortion());

        for (int i = 0; i < kmeans.k(); i++) {
            System.out.printf("  Cluster %d: %3d muestras, radio %.3f, centro %s%n",
                    i, kmeans.size(i), kmeans.radius(i),
                    Arrays.toString(kmeans.center(i)));
        }

        // Asignar un punto nuevo al cluster mas proximo
        double[] nueva = { 5.9, 3.0, 5.1, 1.8 };
        System.out.printf("La muestra %s cae en el cluster %d%n",
                Arrays.toString(nueva), kmeans.predict(nueva));

        Utiles.separador("2. Comparacion con las clases reales");

        // Indices externos: comparan dos particiones sin importar como se
        // numeren los clusters.
        System.out.printf("Rand index          = %.4f%n", RandIndex.of(especieReal, grupos));
        System.out.printf("Rand ajustado (ARI) = %.4f%n", AdjustedRandIndex.of(especieReal, grupos));
        System.out.printf("NMI (sqrt)          = %.4f%n",
                NormalizedMutualInformation.sqrt(especieReal, grupos));

        Utiles.separador("3. Eleccion de k: metodo del codo");

        System.out.printf("%4s %14s%n", "k", "DISTORSION");
        for (int k = 2; k <= 8; k++) {
            CentroidClustering<double[], double[]> modelo = KMeans.fit(x, k, 100);
            System.out.printf("%4d %14.3f%n", k, modelo.distortion());
        }
        System.out.println("Se busca el 'codo': el punto a partir del cual añadir");
        System.out.println("clusters ya no reduce apreciablemente la distorsion.");

        Utiles.separador("4. Eleccion automatica de k: X-means y G-means");

        // Parten de un cluster y van dividiendo mientras mejore el criterio BIC
        // (X-means) o mientras los subgrupos no sean gaussianos (G-means).
        CentroidClustering<double[], double[]> xmeans = XMeans.fit(x, 8, 100);
        System.out.printf("X-means estima k = %d%n", xmeans.k());
        System.out.printf("  ARI frente a las especies reales = %.4f%n",
                AdjustedRandIndex.of(especieReal, xmeans.group()));

        // GMeans.fit(x, kmax, maxIter) tiene exactamente la misma firma.

        Utiles.separador("5. DBSCAN (clustering basado en densidad)");

        // fit(datos, minPts, radio). No hay que fijar k: el algoritmo lo
        // descubre, y ademas identifica ruido.
        DBSCAN<double[]> dbscan = DBSCAN.fit(x, 5, 0.8);
        System.out.printf("Clusters encontrados: %d%n", dbscan.k());

        int[] etiquetasDbscan = dbscan.group();
        long ruido = Arrays.stream(etiquetasDbscan).filter(g -> g == Clustering.OUTLIER).count();
        System.out.printf("Puntos marcados como ruido: %d%n", ruido);

        for (int i = 0; i < dbscan.k(); i++) {
            System.out.printf("  Cluster %d: %d muestras%n", i, dbscan.size(i));
        }

        Utiles.separador("6. Clustering jerarquico");

        // Primero se construye el criterio de enlace sobre la matriz de
        // distancias, despues se corta el dendrograma a la altura deseada.
        HierarchicalClustering jerarquico = HierarchicalClustering.fit(WardLinkage.of(x));
        int[] particion3 = jerarquico.partition(3);

        System.out.printf("Corte en 3 clusters, ARI = %.4f%n",
                AdjustedRandIndex.of(especieReal, particion3));

        // Otros criterios de enlace disponibles en smile.clustering.linkage:
        //   SingleLinkage, CompleteLinkage, UPGMALinkage (media),
        //   UPGMCLinkage, WPGMALinkage, WPGMCLinkage.
        // Tambien se puede cortar por altura: jerarquico.partition(altura).

        Utiles.separador("7. Efecto del escalado");

        // Los algoritmos basados en distancias son sensibles a la escala de
        // las variables. Aqui iris esta en cm en las cuatro columnas, pero con
        // variables heterogeneas (euros, años, metros) hay que estandarizar
        // ANTES de agrupar. Comprobacion rapida:
        double[][] xEstandarizado = MathEx.clone(x);
        MathEx.standardize(xEstandarizado);

        CentroidClustering<double[], double[]> kmeansEstandar = KMeans.fit(xEstandarizado, 3, 100);
        System.out.printf("ARI sin estandarizar = %.4f%n", AdjustedRandIndex.of(especieReal, grupos));
        System.out.printf("ARI estandarizando   = %.4f%n",
                AdjustedRandIndex.of(especieReal, kmeansEstandar.group()));

        // Otros algoritmos del paquete smile.clustering:
        //   KMedoids, KModes (variables categoricas), CLARANS,
        //   HDBSCAN, DENCLUE, MEC, SIB, SpectralClustering,
        //   DeterministicAnnealing.
    }
}
