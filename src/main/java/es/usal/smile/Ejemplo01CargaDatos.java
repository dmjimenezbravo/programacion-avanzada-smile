package es.usal.smile;
import java.util.Arrays;

import smile.data.DataFrame;
import smile.data.type.StructField;
import smile.io.Read;

/**
 * Ejemplo 1: carga y exploracion de datos con {@link DataFrame}.
 *
 * <p>El {@code DataFrame} de SMILE es columnar y tipado: cada columna tiene un
 * {@code DataType} (Double, Int, String...) y opcionalmente una {@code Measure}
 * (nominal, ordinal) que indica que es una variable categorica.</p>
 */
public class Ejemplo01CargaDatos {

    public static void main(String[] args) throws Exception {

        Utiles.separador("1. Lectura de un CSV con cabecera");

        // OJO: por defecto CSVFormat.DEFAULT NO lee la cabecera y las columnas
        // se llamarian V1, V2, V3... Hay que pedirlo explicitamente.
        // El formato se pasa como pares clave=valor separados por comas:
        //   header=true, delimiter=\t, quote=", escape=\, comment=#
        DataFrame iris = Read.csv("data/iris.csv", "header=true");
        System.out.println(iris);

        // Otros formatos disponibles:
        //   Read.arff(ruta)      Read.json(ruta)     Read.parquet(ruta)
        //   Read.arrow(ruta)     Read.avro(ruta)     Read.libsvm(ruta)
        //   Read.data(ruta)      -> infiere el formato por la extension

        Utiles.separador("2. Esquema inferido");

        System.out.printf("Dimensiones: %d filas x %d columnas%n", iris.nrow(), iris.ncol());
        System.out.printf("%-16s %-18s %s%n", "COLUMNA", "TIPO", "MEDIDA");
        for (StructField campo : iris.schema().fields()) {
            System.out.printf("%-16s %-18s %s%n", campo.name(), campo.dtype(), campo.measure());
        }

        Utiles.separador("3. Estadisticos descriptivos");

        // describe() devuelve otro DataFrame con count, media, desviacion,
        // minimo, cuartiles y maximo de cada columna.
        System.out.println(iris.describe());

        Utiles.separador("4. Acceso a columnas y celdas");

        double[] longitudSepalo = iris.column("sepal_length").toDoubleArray();
        String[] especies = iris.column("species").toStringArray();

        System.out.printf("sepal_length[0] = %.1f%n", longitudSepalo[0]);
        System.out.printf("species[0]      = %s%n", especies[0]);
        System.out.printf("celda (0,0)     = %.1f%n", iris.getDouble(0, 0));
        System.out.printf("fila 0 completa = %s%n", iris.get(0));

        Utiles.separador("5. Seleccion, proyeccion y filtrado");

        // Quedarse con algunas columnas
        System.out.println(iris.select("petal_length", "petal_width", "species").head(5));

        // Eliminar columnas
        System.out.println(iris.drop("species").head(3));

        // Filtrar filas: operadores =, !=, <, <=, >, >=, contains, startswith, endswith
        DataFrame grandes = iris.filter("sepal_length", ">", "7.0");
        System.out.printf("Flores con sepalo > 7 cm: %d%n", grandes.nrow());

        DataFrame virginica = iris.filter("species", "=", "virginica");
        System.out.printf("Muestras de virginica: %d%n", virginica.nrow());

        // Rango de filas y muestreo aleatorio
        System.out.println(iris.slice(0, 3));
        System.out.println(iris.sample(4));

        Utiles.separador("6. De texto a nominal: factorize()");

        // Los algoritmos supervisados necesitan la clase como entero con una
        // escala nominal asociada. factorize() convierte las columnas de texto
        // (o las indicadas) en columnas int con NominalScale.
        DataFrame irisNominal = iris.factorize("species");
        StructField clase = irisNominal.schema().field("species");
        System.out.println("Tipo tras factorize : " + clase.dtype());
        System.out.println("Medida asociada     : " + clase.measure());

        int[] y = irisNominal.column("species").toIntArray();
        System.out.println("Primeras etiquetas  : " + Arrays.toString(Arrays.copyOf(y, 10)));

        Utiles.separador("7. Del DataFrame a la matriz double[][]");

        // Muchos algoritmos (KNN, SVM, KMeans, PCA...) trabajan con arrays.
        double[][] x = irisNominal.drop("species").toArray();
        System.out.printf("Matriz de %d x %d%n", x.length, x[0].length);
        System.out.println("Primera fila: " + Arrays.toString(x[0]));

        Utiles.separador("8. Construir un DataFrame en memoria");

        double[][] datos = { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } };
        DataFrame df = DataFrame.of(datos, "a", "b");
        System.out.println(df);

        // Tambien: DataFrame.of(int[][], nombres), DataFrame.of(Clase.class, lista)
        // y DataFrame.of(ResultSet) para leer directamente de JDBC.

        Utiles.separador("9. Valores ausentes");

        DataFrame sucias = Read.csv("data/viviendas_sucias.csv", "header=true");
        System.out.println(sucias.head(8));
        System.out.printf("Filas totales: %d - tras dropna(): %d%n",
                sucias.nrow(), sucias.dropna().nrow());
        // Alternativa rapida: sucias.fillna(0.0). Para imputacion seria,
        // ver el Ejemplo02Preprocesado.
    }
}
