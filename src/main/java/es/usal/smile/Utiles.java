package es.usal.smile;

import java.util.Arrays;

import smile.data.DataFrame;
import smile.math.MathEx;
import smile.util.Index;

/**
 * Utilidades comunes a todos los ejemplos: particiones entrenamiento/test y
 * formato de salida por consola.
 *
 * <p>SMILE no incluye un {@code train_test_split} al estilo de scikit-learn:
 * lo habitual es permutar los indices con {@link MathEx#permutate(int)} y
 * quedarse con dos trozos. Fijando la semilla con {@code MathEx.setSeed(...)}
 * los resultados son reproducibles.</p>
 */
public final class Utiles {

    private Utiles() {
        // clase de utilidades: no instanciable
    }

    /** Particion de un DataFrame en entrenamiento y test. */
    public record Particion(DataFrame train, DataFrame test) { }

    /** Particion de matrices de caracteristicas y etiquetas. */
    public record ParticionArrays(double[][] xTrain, int[] yTrain, double[][] xTest, int[] yTest) { }

    /**
     * Divide un DataFrame de forma aleatoria.
     *
     * @param datos            conjunto completo
     * @param proporcionTrain  fraccion de filas para entrenamiento, p. ej. 0.7
     */
    public static Particion split(DataFrame datos, double proporcionTrain) {
        int n = datos.nrow();
        int[] permutacion = MathEx.permutate(n);
        int corte = (int) Math.round(n * proporcionTrain);

        int[] indicesTrain = Arrays.copyOfRange(permutacion, 0, corte);
        int[] indicesTest  = Arrays.copyOfRange(permutacion, corte, n);

        // DataFrame.get(Index) selecciona filas por indice.
        return new Particion(datos.get(Index.of(indicesTrain)),
                             datos.get(Index.of(indicesTest)));
    }

    /** Divide una matriz de caracteristicas y su vector de etiquetas. */
    public static ParticionArrays split(double[][] x, int[] y, double proporcionTrain) {
        int n = x.length;
        int[] permutacion = MathEx.permutate(n);
        int corte = (int) Math.round(n * proporcionTrain);

        double[][] xTrain = new double[corte][];
        int[] yTrain = new int[corte];
        double[][] xTest = new double[n - corte][];
        int[] yTest = new int[n - corte];

        for (int i = 0; i < corte; i++) {
            xTrain[i] = x[permutacion[i]];
            yTrain[i] = y[permutacion[i]];
        }
        for (int i = corte; i < n; i++) {
            xTest[i - corte] = x[permutacion[i]];
            yTest[i - corte] = y[permutacion[i]];
        }

        return new ParticionArrays(xTrain, yTrain, xTest, yTest);
    }

    /** Imprime un separador con titulo para que la salida sea legible. */
    public static void separador(String titulo) {
        System.out.println();
        System.out.println("=".repeat(72));
        System.out.println("  " + titulo);
        System.out.println("=".repeat(72));
    }
}
