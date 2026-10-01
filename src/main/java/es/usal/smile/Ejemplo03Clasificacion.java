package es.usal.smile;

import java.nio.file.Path;

import smile.classification.KNN;
import smile.classification.LDA;
import smile.classification.LogisticRegression;
import smile.classification.RandomForest;
import smile.data.CategoricalEncoder;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.measure.NominalScale;
import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;
import smile.data.vector.StringVector;
import smile.io.Read;
import smile.io.Write;
import smile.math.MathEx;
import smile.validation.CrossValidation;
import smile.validation.metric.Accuracy;
import smile.validation.metric.ConfusionMatrix;

/**
 * Ejemplo 3: clasificacion supervisada.
 *
 * <p>SMILE ofrece dos estilos de API:</p>
 * <ul>
 *   <li><b>Con formula</b>: {@code fit(Formula, DataFrame)}. El modelo recuerda
 *       el esquema y predice directamente sobre un {@code Tuple}. Es el estilo
 *       de los modelos basados en arboles, que aceptan variables categoricas.</li>
 *   <li><b>Con arrays</b>: {@code fit(double[][] x, int[] y)}. Lo usan KNN, SVM,
 *       LDA, regresion logistica, redes neuronales...</li>
 * </ul>
 */
public class Ejemplo03Clasificacion {

    public static void main(String[] args) throws Exception {

        // Semilla global: hace reproducibles las particiones y los modelos
        // estocasticos (random forest, validacion cruzada...).
        MathEx.setSeed(42);

        DataFrame iris = Read.csv("data/iris.csv", "header=true").factorize("species");

        // "species ~ ." : la clase es species, el resto son predictoras.
        Formula formula = Formula.lhs("species");

        Utiles.Particion particion = Utiles.split(iris, 0.7);
        DataFrame train = particion.train();
        DataFrame test = particion.test();
        System.out.printf("Entrenamiento: %d filas | Test: %d filas%n", train.nrow(), test.nrow());

        // Nombres legibles de las clases, a partir de la escala nominal.
        NominalScale escala = (NominalScale) iris.schema().field("species").measure();

        Utiles.separador("1. Random Forest (API con formula)");

        // Los hiperparametros van en un record Options. Constructores abreviados:
        //   new Options(ntrees)
        //   new Options(ntrees, mtry, maxDepth, maxNodes, nodeSize)
        RandomForest.Options opciones = new RandomForest.Options(200);
        RandomForest bosque = RandomForest.fit(formula, train, opciones);

        // Metricas out-of-bag: el random forest se autoevalua con las muestras
        // que quedan fuera de cada bootstrap, sin necesidad de conjunto aparte.
        System.out.println("Metricas OOB: " + bosque.metrics());

        System.out.println("Importancia de variables:");
        double[] importancia = bosque.importance();
        String[] nombres = bosque.schema().names();
        for (int i = 0; i < importancia.length; i++) {
            System.out.printf("  %-16s %10.3f%n", nombres[i], importancia[i]);
        }

        Utiles.separador("2. Evaluacion sobre el conjunto de test");

        int[] yTest = formula.y(test).toIntArray();
        int[] prediccion = new int[test.nrow()];
        for (int i = 0; i < test.nrow(); i++) {
            prediccion[i] = bosque.predict(test.get(i));   // predict(Tuple)
        }

        System.out.printf("Accuracy = %.2f%%%n", 100.0 * Accuracy.of(yTest, prediccion));
        System.out.println("Matriz de confusion:");
        System.out.println(ConfusionMatrix.of(yTest, prediccion));

        Utiles.separador("3. Prediccion blanda (probabilidades a posteriori)");

        double[] posteriori = new double[escala.size()];
        int clase = bosque.predict(test.get(0), posteriori);
        System.out.printf("Clase predicha: %s%n", escala.level(clase));
        for (int k = 0; k < posteriori.length; k++) {
            System.out.printf("  P(%-12s) = %.4f%n", escala.level(k), posteriori[k]);
        }

        Utiles.separador("4. Modelos que trabajan con arrays");

        double[][] xTrain = formula.x(train).toArray();
        int[] yTrain = formula.y(train).toIntArray();
        double[][] xTest = formula.x(test).toArray();

        KNN<double[]> knn = KNN.fit(xTrain, yTrain, 5);
        System.out.printf("KNN (k=5)            accuracy = %.2f%%%n",
                100.0 * Accuracy.of(yTest, knn.predict(xTest)));

        // Options(lambda, tolerancia, maxIteraciones). lambda regulariza L2.
        LogisticRegression logistica = LogisticRegression.fit(xTrain, yTrain,
                new LogisticRegression.Options(0.1, 1E-5, 500));
        System.out.printf("Regresion logistica  accuracy = %.2f%%%n",
                100.0 * Accuracy.of(yTest, logistica.predict(xTest)));

        // LDA hace una descomposicion en autovalores que SMILE delega en
        // BLAS/LAPACK nativo (OpenBLAS). Si la libreria nativa no esta en el
        // PATH, la carga falla con un Error (no una Exception) al primer uso;
        // lo capturamos para que el resto del ejemplo pueda seguir.
        try {
            LDA lda = LDA.fit(xTrain, yTrain);
            System.out.printf("LDA                  accuracy = %.2f%%%n",
                    100.0 * Accuracy.of(yTest, lda.predict(xTest)));
        } catch (Throwable fallo) {
            System.out.println("LDA no disponible: falta la libreria nativa OpenBLAS/LAPACK. "
                    + "Instala OpenBLAS y anade su carpeta 'bin' al PATH para habilitarla.");
        }

        // Otros clasificadores del paquete smile.classification:
        //   DecisionTree, AdaBoost, GradientTreeBoost, QDA, RDA, FLD,
        //   NaiveBayes, DiscreteNaiveBayes, Maxent, RBFNetwork, MLP,
        //   SVM (+ OneVersusOne / OneVersusRest para multiclase).

        Utiles.separador("5. Validacion cruzada");

        // La estimacion honesta del rendimiento: k particiones, k modelos.
        var resultado = CrossValidation.classification(10, formula, iris,
                (f, datos) -> RandomForest.fit(f, datos, opciones));

        System.out.println(resultado);
        System.out.printf("Accuracy media = %.4f (desv. %.4f)%n",
                resultado.avg().accuracy(), resultado.std().accuracy());

        // Variantes utiles:
        //   CrossValidation.stratify(10, ...)   mantiene la proporcion de clases
        //   CrossValidation.classification(5, 10, ...)  5 repeticiones de 10-fold
        //   LOOCV.classification(x, y, trainer)  leave-one-out
        //   Bootstrap.classification(...)

        // Con arrays la firma es analoga:
        var cvKnn = CrossValidation.classification(10, xTrain, yTrain,
                (x, y) -> KNN.fit(x, y, 5));
        System.out.printf("KNN 10-fold accuracy = %.4f%n", cvKnn.avg().accuracy());

        Utiles.separador("6. Persistencia del modelo");

        Path ruta = Path.of("modelo-rf.sml");
        Write.object(bosque, ruta);

        RandomForest recuperado = (RandomForest) Read.object(ruta);
        System.out.printf("Modelo recargado, prediccion de la fila 0: %s%n",
                escala.level(recuperado.predict(test.get(0))));

        // Read.object aplica una lista blanca de clases permitidas al
        // deserializar, para evitar los problemas clasicos de seguridad
        // de la serializacion de Java.

        Utiles.separador("7. Clasificar una instancia nueva (fuera del dataset)");

        // --- API con formula: no hace falta codificar nada a mano ---
        // Se construye un Tuple con el esquema de las predictoras del modelo
        // (bosque.schema(), las 4 columnas numericas de iris) y SMILE resuelve
        // los campos por nombre. Si hubiera columnas categoricas, tambien se
        // resolverian solas: el modelo ya sabe como codificarlas porque las vio
        // al entrenar.
        Tuple flor = Tuple.of(bosque.schema(), new double[] { 5.9, 3.0, 4.2, 1.5 });
        System.out.printf("Random Forest predice: %s%n", escala.level(bosque.predict(flor)));

        // --- API con arrays: la instancia es un double[] normal y corriente,
        // con las columnas en el mismo orden que xTrain ---
        double[] florArray = { 5.9, 3.0, 4.2, 1.5 };
        System.out.printf("KNN predice:            %s%n", escala.level(knn.predict(florArray)));

        // Iris no tiene predictoras categoricas (solo la clase), asi que para
        // ver los encoders con valores no numericos de verdad usamos otro
        // dataset: ver seccion 8.

        Utiles.separador("8. Encoders con un dataset que tiene predictoras categoricas");

        // prestamos.csv es sintetico (aprobacion de un prestamo a partir de
        // datos socioeconomicos) y, a diferencia de iris, mezcla columnas
        // numericas con tres columnas de texto: tipo_empleo, vivienda e
        // historial_crediticio.
        DataFrame prestamosRaw = Read.csv("data/prestamos.csv", "header=true");

        // La instancia nueva, tal como la introduciria un usuario: valores en
        // crudo, con las categoricas como texto. Se concatena ANTES de
        // factorizar para que quede codificada con exactamente los mismos
        // niveles que el entrenamiento (factorize() asigna los codigos por
        // orden alfabetico de los valores que ve).
        DataFrame instanciaRaw = new DataFrame(
                new IntVector("edad", new int[] { 34 }),
                new DoubleVector("ingresos_anuales", new double[] { 42000.0 }),
                new DoubleVector("deuda_actual", new double[] { 8000.0 }),
                new StringVector("tipo_empleo", new String[] { "autonomo" }),
                new StringVector("vivienda", new String[] { "alquiler" }),
                new StringVector("historial_crediticio", new String[] { "regular" }),
                // "aprobado" es justo lo que queremos predecir: se pone un
                // valor cualquiera de los que ya existen solo para que el
                // esquema encaje al concatenar (si no, factorize() veria un
                // nivel nuevo y desplazaria los codigos del entrenamiento).
                new StringVector("aprobado", new String[] { "no" }));

        DataFrame combinado = prestamosRaw.concat(instanciaRaw)
                .factorize("tipo_empleo", "vivienda", "historial_crediticio", "aprobado");

        DataFrame prestamos = combinado.slice(0, prestamosRaw.nrow());
        DataFrame instanciaDf = combinado.slice(prestamosRaw.nrow(), combinado.nrow());

        Formula formulaPrestamos = Formula.lhs("aprobado");
        NominalScale escalaAprobado = (NominalScale) prestamos.schema().field("aprobado").measure();

        Utiles.Particion particionPrestamos = Utiles.split(prestamos, 0.75);
        DataFrame trainP = particionPrestamos.train();
        DataFrame testP = particionPrestamos.test();
        int[] yTestP = formulaPrestamos.y(testP).toIntArray();

        // --- Random Forest: acepta las columnas categoricas tal cual, sin
        // codificarlas a mano (igual que con iris) ---
        RandomForest bosquePrestamos = RandomForest.fit(formulaPrestamos, trainP,
                new RandomForest.Options(200));
        int[] predP = new int[testP.nrow()];
        for (int i = 0; i < testP.nrow(); i++) {
            predP[i] = bosquePrestamos.predict(testP.get(i));
        }
        System.out.printf("Random Forest (categoricas nativas)  accuracy = %.2f%%%n",
                100.0 * Accuracy.of(yTestP, predP));

        // --- KNN: necesita numeros, asi que comparamos como codificar las
        // tres columnas de texto ---
        //   LEVEL   -> un entero por nivel (0, 1, 2...). Introduce un orden y
        //              una distancia entre niveles que no existen de verdad
        //              (para KNN, "bueno" y "malo" acaban mas o menos cerca o
        //              lejos segun les toque el numero, no segun su sentido).
        //   ONE_HOT -> una columna binaria por nivel. Todas las categorias
        //              quedan a la misma distancia entre si: es lo correcto
        //              para modelos de distancia o de combinacion lineal.
        int[] yTrainP = formulaPrestamos.y(trainP).toIntArray();

        double[][] xTrainLevel = formulaPrestamos.x(trainP).toArray(false, CategoricalEncoder.LEVEL);
        double[][] xTestLevel = formulaPrestamos.x(testP).toArray(false, CategoricalEncoder.LEVEL);
        KNN<double[]> knnLevel = KNN.fit(xTrainLevel, yTrainP, 5);
        System.out.printf("KNN con LEVEL                        accuracy = %.2f%%%n",
                100.0 * Accuracy.of(yTestP, knnLevel.predict(xTestLevel)));

        double[][] xTrainOneHot = formulaPrestamos.x(trainP).toArray(false, CategoricalEncoder.ONE_HOT);
        double[][] xTestOneHot = formulaPrestamos.x(testP).toArray(false, CategoricalEncoder.ONE_HOT);
        KNN<double[]> knnOneHot = KNN.fit(xTrainOneHot, yTrainP, 5);
        System.out.printf("KNN con ONE_HOT                      accuracy = %.2f%%%n",
                100.0 * Accuracy.of(yTestP, knnOneHot.predict(xTestOneHot)));

        // Aqui LEVEL y ONE_HOT dan practicamente la misma accuracy, y las dos
        // se quedan muy por debajo del Random Forest: sin estandarizar,
        // ingresos_anuales y deuda_actual (decenas de miles) dominan la
        // distancia euclidea de KNN frente a las columnas categoricas, tenga
        // estas 1 o 3 dimensiones. El encoder esta haciendo bien su trabajo
        // -las dos codificaciones son numericamente correctas y se puede
        // comprobar imprimiendo xTrainLevel/xTrainOneHot-; lo que falta es el
        // Standardizer del Ejemplo 2 antes de entrenar.

        // --- Clasificar la instancia nueva ---
        System.out.printf("%nRandom Forest predice para la instancia nueva:  %s%n",
                escalaAprobado.level(bosquePrestamos.predict(instanciaDf.get(0))));

        // Con KNN hay que pasar la instancia por el mismo pipeline de
        // codificacion que xTrain (aqui, ONE_HOT) antes de predecir.
        double[] instanciaOneHot = formulaPrestamos.x(instanciaDf)
                .toArray(false, CategoricalEncoder.ONE_HOT)[0];
        System.out.printf("KNN (ONE_HOT) predice para la instancia nueva:  %s%n",
                escalaAprobado.level(knnOneHot.predict(instanciaOneHot)));
    }
}
