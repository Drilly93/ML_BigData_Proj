# PySpark eALS : Distributed Recommender System for Implicit Feedback

Ce projet propose une implémentation distribuée *from scratch* en PySpark de l'algorithme **eALS (element-wise Alternating Least Squares)**, conçu pour les systèmes de recommandation basés sur des retours implicites (clics, vues, achats).

L’implémentation est optimisée pour traiter des jeux de données massifs (comme le Yelp Academic Dataset) en combinant la scalabilité horizontale de Spark et la vitesse de calcul vectoriel de NumPy.

## Référence

Basé sur l’article de recherche :
*Fast Matrix Factorization for Online Recommendation with Implicit Feedback* (He et al., SIGIR 2016).

---

## Caractéristiques et optimisations Big Data

Contrairement à l’ALS classique de Spark MLlib, qui utilise une pondération uniforme pour les données manquantes, cette implémentation introduit le **Popularity-aware Item Weighting** tout en évitant une explosion de la complexité temporelle.

### Optimisations architecturales PySpark

* **NumPy via mapPartitions**
  L’algèbre linéaire (matrices de caches $S^p$ et $S^q$) est exécutée en C/Fortran dans la mémoire locale des exécuteurs, réduisant fortement le coût de sérialisation Python (Py4J).

* **Checkpointing local**
  Coupure du graphe d’exécution (lineage) à chaque itération pour éviter les erreurs de type StackOverflow et Out Of Memory.

* **Broadcast joins**
  Diffusion en mémoire des poids de popularité ($c_i$) et des matrices globales ($K \times K$) afin d’éliminer les shuffles réseau coûteux.

* **Tungsten SQL trick**
  Le calcul final du produit scalaire est généré dynamiquement sous forme de requête SQL native, exécutée en C++ via le moteur Tungsten de Spark.

* **Smart partitioning**
  Repartitionnement des DataFrames par `user_idx` et `item_idx` pour garantir la localité des données lors de l’optimisation alternée.

---

## Architecture du projet

```
├── data_prep.py          # Classe ImplicitDataPreprocessor (filtrage k-core, indexation)
├── eALS_pyspark.py       # Classe PySpark_eALS (cœur algorithmique, mathématiques, télémétrie)
├── main.py               # Orchestrateur (split train/test, lancement de l'entraînement)
├── requirements.txt      # Dépendances du projet
└── README.md             # Documentation
```

---

## Évaluation et protocole expérimental

L’évaluation suit les standards académiques pour les systèmes de recommandation implicites.

* **Train/Test split (Leave-One-Out)**
  Pour chaque utilisateur, une interaction est masquée et utilisée comme vérité terrain.

* **Negative sampling (1 vs 99)**
  L’item caché est mélangé avec 99 items non-interagis.

* **Métriques**

  * Hit Ratio @10 (HR@10)
  * NDCG @10

Le modèle doit classer correctement l’item pertinent parmi 100 candidats.

---

## Installation et prérequis

### 1. Environnement Python

Utilisation recommandée de Conda avec Python 3.11 :

```bash
conda create -n bigdata_env python=3.11 -y
conda activate bigdata_env
pip install -r requirements.txt
```

### 2. Configuration système (Windows)

Pour une exécution locale sous Windows :

* Installer Java (version 8, 11 ou 17) et définir `JAVA_HOME`
* Installer les binaires Hadoop (`winutils.exe`, `hadoop.dll`)
* Définir la variable `HADOOP_HOME` (exemple : `C:\hadoop\bin`)

---

## Utilisation

1. Télécharger le dataset Yelp (ou tout autre dataset de feedback implicite) au format JSON
2. Modifier la variable `chemin_dataset_yelp` dans `main.py`
3. Lancer l'entraînement :

```bash
python main.py
```

---

## Tableau de bord (télémétrie)

Pendant l’entraînement, un tableau de bord s’affiche à chaque itération :

```
============================================================
 DÉMARRAGE MOTEUR eALS - 10 ITÉRATIONS (K=64)
============================================================

[ INFO ] Dataset : 15423 Utilisateurs | 8502 Items
------------------------------------------------------------

[ ITERATION 1/10 TERMINEE en 12.4s ]
 Qualité   | HR@10: 0.1240  | NDCG@10: 0.0581
 Calculs   | eALS: 98.0 Mops | Classique: 6271.4 Mops
 Gain      | Facteur d'accélération mathématique : x64
 Réseau    | Shuffles Spark esquivés via NumPy : 95%
------------------------------------------------------------
```

---

## Hyperparamètres par défaut

* **K (facteurs latents)** : 64
* **λ (régularisation L2)** : 0.01
* **c₀ (poids de confiance global)** : 512.0
* **α (exposant de popularité)** : 0.4
* **Filtrage k-core** : 10

---

## Auteur

Développé dans le cadre d’un projet de Machine Learning et Big Data.

Mathéo Quatreboeufs
Implémentation algorithmique, ingénierie PySpark et optimisation mathématique.


