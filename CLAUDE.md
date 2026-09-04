# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Teaching material for "Programación Avanzada" (USAL): six didactic Java examples using the
**SMILE** machine learning library (v6.3.0) — classification, regression, clustering, and PCA.
Written in Spanish (code comments, output, docs). The authoritative reference for SMILE's API
in this repo is [src/GUIA-SMILE.md](src/GUIA-SMILE.md) — read it before making non-trivial
changes to any example, since it documents version-specific gotchas (see below) that are easy
to get wrong from general SMILE knowledge or outdated tutorials.

## Commands

```bash
mvn compile
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo01CargaDatos
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo02Preprocesado
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo03Clasificacion
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo04Regresion
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo05Clustering
mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo06PCA
```

There is no test suite — this is example/demo code, verified by running it and reading the
console output. Always run `mvn compile exec:java -Dexec.mainClass=...` from the repository
root, since the examples load CSVs from `data/` via relative paths (`data/iris.csv`,
`data/viviendas.csv`).

Requires **JDK 25** (`maven.compiler.release` in [pom.xml](pom.xml)) — SMILE 6.x's API (record-based
`Options`, `DataFrame` as a class rather than interface) needs it. If only JDK 21 is available,
SMILE 6.x won't compile; downgrading to SMILE 4.x would be required and would touch API details
throughout every example.

## Architecture

- One `main()` per topic in `src/main/java/es/usal/smile/`, numbered in teaching order
  (`Ejemplo01CargaDatos` → `Ejemplo06PCA`). Each is self-contained and run independently via
  `exec:java`; they are not wired together.
- [Utiles.java](src/main/java/es/usal/smile/Utiles.java) holds the only shared code: train/test
  splitting (`Utiles.split`, for both `DataFrame` and raw `double[][]`/`int[]` arrays, since SMILE
  has no built-in `train_test_split`) and `Utiles.separador(titulo)` for readable console section
  headers. New shared helpers belong here, not duplicated per example.
- Data lives in `data/*.csv`: `iris.csv` (classic multiclass dataset, used across classification,
  clustering and PCA examples) and `viviendas.csv` (synthetic housing data generated from a known
  linear relation + Gaussian noise — see the formula documented at the top of
  `Ejemplo04Regresion.java` — used to check that OLS recovers the true coefficients and that a
  random forest underperforms OLS when the generating process is actually linear).
  `viviendas_sucias.csv` is a "dirty" variant, currently unused by any example.
- Every example calls `MathEx.setSeed(42)` before anything stochastic (splits, random forests,
  cross-validation) for reproducibility — carry this over in new/modified examples.
- Two coexisting SMILE API styles show up deliberately across examples: formula + `DataFrame`
  (`Model.fit(Formula, DataFrame, Options)`, used by tree-based models that accept categorical
  columns unencoded) vs. raw arrays (`Model.fit(double[][] x, int[] y)`, used by KNN, SVM,
  LDA/QDA, logistic regression, MLP). Match whichever style the surrounding example already uses.

## SMILE-specific pitfalls (see §9 of the guide for the full list)

These recur throughout the examples and are easy to reintroduce when editing:

- `Read.csv(path, "header=true")` — the header flag is **not** the default; without it columns
  come back named `V1`, `V2`, ... `Read.csv` and other `Read.*` methods live in `smile.io.Read`.
- CSV-sourced label/categorical columns must be run through `.factorize("colname")` before use in
  classification — otherwise `toIntArray()`/casts on that column fail. ARFF files don't need this.
- `PCA` is in `smile.feature.extraction`, not `smile.projection` (that was the 2.x location and
  still shows up in outdated tutorials/docs).
- Preprocessing transforms (`Standardizer`, `Scaler`, etc.) follow `fit(train)` / `apply(...)`
  (like sklearn's fit/transform): fit **only** on the training split, then apply to both train and
  test — fitting on the full dataset before splitting is a data leak.
- Hyperparameters in SMILE 6.x are nested `record` types (e.g. `new RandomForest.Options(200)`),
  not `Properties` — `Properties`-based examples online are for SMILE 3.x.
- Clustering algorithms are distance-based; standardize features first when they're on different
  scales.
