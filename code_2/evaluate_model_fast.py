import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Import de nos classes personnalisées
from eALS_pyspark import PySpark_eALS
from data_prep import ImplicitDataPreprocessor

# --- FIX CRUCIAL POUR PYTHON 3.11 ET CONDA ---
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def prepare_train_test(df_interactions: DataFrame, num_test_users: int = 500) -> tuple[DataFrame, DataFrame]:
    """
    Sépare le dataset en Train/Test en utilisant la méthode "Leave-One-Out".
    """
    print(f"\n--- Création du set d'évaluation ({num_test_users} utilisateurs) ---")
    
    df_users = df_interactions.select("user_idx").distinct()
    total_users = df_users.count()
    
    fraction = min(1.0, (num_test_users * 1.5) / total_users)
    sampled_users = df_users.sample(withReplacement=False, fraction=fraction, seed=42) \
                            .limit(num_test_users)
    
    df_target_interactions = df_interactions.join(sampled_users, on="user_idx", how="inner")
    
    window_spec = Window.partitionBy("user_idx").orderBy(F.rand(seed=42))
    
    df_test = df_target_interactions.withColumn("random_rank", F.row_number().over(window_spec)) \
                                    .filter(F.col("random_rank") == 1) \
                                    .drop("random_rank")
    
    df_test = df_test.cache()
    actual_test_users = df_test.count()
    print(f"-> {actual_test_users} interactions cachées formées pour l'évaluation (df_test).")

    df_train = df_interactions.join(df_test, on=["user_idx", "item_idx"], how="left_anti")
    df_train = df_train.repartition("user_idx").cache()
    train_count = df_train.count()
    
    print(f"-> {train_count} interactions restantes pour l'entraînement (df_train).")
    return df_train, df_test

def main():
    # 1. Initialisation de Spark (Avec Optimisations)
    print("Initialisation de la SparkSession...")
    spark = SparkSession.builder \
        .appName("eALS_Yelp_Training") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    # Configuration vitale pour le `.localCheckpoint()`
    checkpoint_dir = "/data/spark_checkpoints_eals"
    os.makedirs(checkpoint_dir, exist_ok=True)
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    spark.sparkContext.setLogLevel("ERROR") # Pour éviter le spam dans la console

    # 2. Chargement du vrai dataset Yelp
    chemin_dataset_yelp = r"c:\Users\USER\Documents\data\Yelp-JSON\Yelp JSON\yelp_dataset\yelp_academic_dataset_review.json"
    
    print(f"\n--- Chargement des données brutes depuis : {chemin_dataset_yelp} ---")
    try:
        df_raw = spark.read.json(chemin_dataset_yelp)
    except Exception as e:
        print(f"ERREUR : Impossible de charger le fichier JSON. Vérifie le chemin !\n{e}")
        spark.stop()
        return

    # 3. Pipeline de préparation des données (Filtrage des utilisateurs < 10 interactions)
    preprocessor = ImplicitDataPreprocessor(user_col="user_id", item_col="business_id")
    df_interactions, df_popularity = preprocessor.transform(df_raw, min_interactions=10)

    # 4. Séparation Train/Test
    df_train, df_test = prepare_train_test(df_interactions, num_test_users=500)

    # 5. Configuration et Lancement du Modèle
    # Paramètres recommandés pour démarrer sur un gros dataset : K=64, 10 itérations
    print("\n--- Initialisation du Modèle eALS ---")
    eals_model = PySpark_eALS(K=64, max_iter=10)
    
    # Démarrage du moteur et de la télémétrie
    eals_model.fit_with_telemetry(df_train, df_test, df_popularity, spark)

    # Optionnel : Sauvegarder le modèle à la fin (les DataFrames P et Q)
    # eals_model.P.write.parquet("model_output/matrice_P.parquet", mode="overwrite")
    # eals_model.Q.write.parquet("model_output/matrice_Q.parquet", mode="overwrite")

    print("\nProcessus terminé avec succès.")
    spark.stop()

if __name__ == "__main__":
    main()