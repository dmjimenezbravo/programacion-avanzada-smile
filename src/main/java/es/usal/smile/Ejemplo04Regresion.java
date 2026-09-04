package es.usal.smile;

import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.math.MathEx;
import smile.regression.ElasticNet;
import smile.regression.GradientTreeBoost;
import smile.regression.LASSO;
import smile.regression.LinearModel;
import smile.regression.OLS;
import smile.regression.RandomForest;
import smile.regression.RidgeRegression;
import smile.tensor.Vector;
import smile.validation.CrossValidation;
import smile.validation.metric.MAD;
import smile.validation.metric.R2;
import smile.validation.metric.RMSE;

/**
 * Ejemplo 4: regresion.
 *
 * <p>El conjunto {@code viviendas.csv} es sintetico y se ha generado con una
 * relacion lineal conocida mas ruido gaussiano, de modo que los coeficientes
 * estimados por minimos cuadrados se pueden comparar con los reales:</p>
 *
 * <pre>
 *   precio = 1.8*superficie + 12*habitaciones + 9*banos
 *            - 1.1*antiguedad - 6.5*distancia_centro + 55 + N(0, 12)
 * </pre>
 */
public class Ejemplo04Regresion {

    public static void main(String[] args) throws Exception {

        MathEx.setSeed(42);

        // Practicamente todo este ejemplo gira en torno a OLS: su fit()
        // resuelve una descomposicion QR que SMILE delega en BLAS/LAPACK
        // nativo (OpenBLAS), igual que LDA en el Ejemplo 3. Si esa libreria
        // no esta en el PATH, falla como un Error (no una Exception) al
        // primer uso, y el resto del ejemplo depende de "ols", asi que no
        // tiene sentido intentar continuar: se captura una sola vez aqui.
        try {
            ejecutar();
        } catch (Throwable fallo) {
            if (esFalloDeLibreriaNativa(fallo)) {
                System.out.println("Este ejemplo no puede continuar: OLS necesita BLAS/LAPACK "
                        + "nativo (OpenBLAS) para su descomposicion QR y no esta disponible en "
                        + "el PATH. Instala OpenBLAS y anade su carpeta 'bin' al PATH para "
                        + "poder ejecutarlo.");
            } else {
                throw fallo;
            }
        }
    }

    /** Busca en la cadena de causas un fallo al cargar una libreria nativa. */
    private static boolean esFalloDeLibreriaNativa(Throwable fallo) {
        for (Throwable causa = fallo; causa != null; causa = causa.getCause()) {
            if (causa instanceof IllegalArgumentException
                    && causa.getMessage() != null
                    && causa.getMessage().startsWith("Cannot open library")) {
                return true;
            }
        }
        return false;
    }

    private static void ejecutar() throws Exception {

        DataFrame viviendas = Read.csv("data/viviendas.csv", "header=true");
        Formula formula = Formula.lhs("precio");

        Utiles.Particion particion = Utiles.split(viviendas, 0.75);
        DataFrame train = particion.train();
        DataFrame test = particion.test();

        Utiles.separador("1. Minimos cuadrados ordinarios (OLS)");

        LinearModel ols = OLS.fit(formula, train);

        // toString() imprime un resumen al estilo de summary() en R:
        // coeficientes, error estandar, estadistico t, p-valor, R2 y test F.
        System.out.println(ols);

        Utiles.separador("2. Interpretacion del modelo lineal");

        System.out.printf("Termino independiente : %.3f  (real: 55)%n", ols.intercept());

        Vector coeficientes = ols.coefficients();
        String[] predictoras = ols.schema().names();
        double[] reales = { 1.8, 12.0, 9.0, -1.1, -6.5 };

        System.out.printf("%-20s %12s %12s%n", "VARIABLE", "ESTIMADO", "REAL");
        for (int i = 0; i < reales.length; i++) {
            System.out.printf("%-20s %12.4f %12.4f%n", predictoras[i], coeficientes.get(i), reales[i]);
        }

        System.out.printf("%nR2 = %.4f | R2 ajustado = %.4f%n", ols.RSquared(), ols.adjustedRSquared());
        System.out.printf("Estadistico F = %.2f (p = %.3e)%n", ols.ftest(), ols.pvalue());
        System.out.printf("Error residual = %.3f con %d grados de libertad%n", ols.error(), ols.df());

        // ttest() devuelve una matriz con [coeficiente, error estandar, t, p-valor]
        // por cada variable, por si se quiere procesar programaticamente.
        double[][] tabla = ols.ttest();
        System.out.printf("p-valor de la primera variable: %.3e%n", tabla[0][3]);

        Utiles.separador("3. Evaluacion sobre test");

        double[] yTest = formula.y(test).toDoubleArray();
        double[] prediccion = ols.predict(test);   // predict(DataFrame) -> double[]

        System.out.printf("RMSE = %.3f%n", RMSE.of(yTest, prediccion));
        System.out.printf("MAD  = %.3f%n", MAD.of(yTest, prediccion));
        System.out.printf("R2   = %.4f%n", R2.of(yTest, prediccion));

        // Prediccion de un caso concreto
        System.out.printf("Precio predicho para la primera vivienda de test: %.1f (real %.1f)%n",
                ols.predict(test.get(0)), yTest[0]);

        Utiles.separador("4. Modelos regularizados");

        // Cuando hay colinealidad o muchas variables, la regularizacion evita
        // el sobreajuste. LASSO ademas anula coeficientes (seleccion implicita).
        // RidgeRegression, a diferencia de LASSO/ElasticNet, no tiene una
        // sobrecarga fit(Formula, DataFrame, Options); el lambda se pasa
        // directamente como double (la version con Options es para ajustar
        // varios lambdas de una vez con un vector).
        LinearModel ridge = RidgeRegression.fit(formula, train, 0.1);
        LinearModel lasso = LASSO.fit(formula, train, new LASSO.Options(0.5));
        LinearModel elastic = ElasticNet.fit(formula, train, new ElasticNet.Options(0.5, 0.1));

        System.out.printf("Ridge       RMSE = %.3f%n", RMSE.of(yTest, ridge.predict(test)));
        System.out.printf("LASSO       RMSE = %.3f%n", RMSE.of(yTest, lasso.predict(test)));
        System.out.printf("ElasticNet  RMSE = %.3f%n", RMSE.of(yTest, elastic.predict(test)));

        Utiles.separador("5. Modelos no lineales basados en arboles");

        RandomForest bosque = RandomForest.fit(formula, train, new RandomForest.Options(300));
        System.out.println("Random Forest OOB: " + bosque.metrics());
        System.out.printf("Random Forest RMSE = %.3f%n", RMSE.of(yTest, bosque.predict(test)));

        System.out.println("Importancia de variables:");
        double[] importancia = bosque.importance();
        for (int i = 0; i < importancia.length; i++) {
            System.out.printf("  %-20s %12.1f%n", predictoras[i], importancia[i]);
        }

        GradientTreeBoost gbt = GradientTreeBoost.fit(formula, train,
                new GradientTreeBoost.Options(300));
        System.out.printf("Gradient Boosting RMSE = %.3f%n", RMSE.of(yTest, gbt.predict(test)));

        // Con datos generados por un modelo lineal, OLS deberia ganar a los
        // arboles: es la mejor forma de que los alumnos vean que "mas complejo"
        // no significa "mejor".

        Utiles.separador("6. Validacion cruzada");

        var cvOls = CrossValidation.regression(10, formula, viviendas,
                (f, datos) -> OLS.fit(f, datos));
        var cvRf = CrossValidation.regression(10, formula, viviendas,
                (f, datos) -> RandomForest.fit(f, datos, new RandomForest.Options(100)));

        System.out.printf("OLS           RMSE medio = %.3f | R2 medio = %.4f%n",
                cvOls.avg().rmse(), cvOls.avg().r2());
        System.out.printf("Random Forest RMSE medio = %.3f | R2 medio = %.4f%n",
                cvRf.avg().rmse(), cvRf.avg().r2());

        // Otros regresores en smile.regression:
        //   RegressionTree, SVM, GaussianProcessRegression, RBFNetwork,
        //   MLP, GAM (modelos aditivos generalizados), GLM.
    }
}
