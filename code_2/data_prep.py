from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from typing import Tuple, Dict

class ImplicitDataPreprocessor:
    """
    Classe dédiée à la préparation, au filtrage et à l'indexation des données 
    pour les algorithmes de recommandation implicite distribués.
    """

    def __init__(self, user_col: str = "user_id", item_col: str = "business_id"):
        """
        Initialise le préprocesseur.
        
        Args:
            user_col (str): Nom de la colonne contenant l'ID utilisateur brut.
            item_col (str): Nom de la colonne contenant l'ID item brut.
        """
        self.user_col = user_col
        self.item_col = item_col
        self.indexer_models: Dict[str, StringIndexerModel] = {}

    def _filter_interactions(self, df: DataFrame, min_interactions: int) -> DataFrame:
        """
        Filtre itérativement (k-core) pour retirer les utilisateurs et items sous le seuil d'activité.
        """
        df.cache()
        current_count = df.count()
        previous_count = -1
        iteration = 1

        print(f"--- Début du filtrage (Seuil: {min_interactions}) ---")
        
        while current_count != previous_count:
            previous_count = current_count

            # Filtrage utilisateurs
            user_counts = df.groupBy(self.user_col).agg(F.count("*").alias("u_count"))
            valid_users = user_counts.filter(F.col("u_count") >= min_interactions).select(self.user_col)
            df = df.join(valid_users, on=self.user_col, how="inner")

            # Filtrage items
            item_counts = df.groupBy(self.item_col).agg(F.count("*").alias("i_count"))
            valid_items = item_counts.filter(F.col("i_count") >= min_interactions).select(self.item_col)
            df = df.join(valid_items, on=self.item_col, how="inner")

            # Sécurité mémoire vitale : coupe le graphe d'exécution
            df = df.localCheckpoint() 
            
            current_count = df.count()
            print(f"Itération {iteration}: {current_count} interactions restantes.")
            iteration += 1

        return df

    def _create_integer_indices(self, df: DataFrame) -> DataFrame:
        """
        Convertit les identifiants textuels en indices entiers contigus (0 à N-1).
        """
        user_indexer = StringIndexer(inputCol=self.user_col, outputCol="user_idx")
        user_model = user_indexer.fit(df)
        df = user_model.transform(df)
        
        item_indexer = StringIndexer(inputCol=self.item_col, outputCol="item_idx")
        item_model = item_indexer.fit(df)
        df = item_model.transform(df)

        self.indexer_models['user'] = user_model
        self.indexer_models['item'] = item_model

        df = df.withColumn("user_idx", F.col("user_idx").cast("integer")) \
               .withColumn("item_idx", F.col("item_idx").cast("integer"))

        return df

    def _compute_item_popularity(self, df: DataFrame) -> DataFrame:
        """
        Calcule la fréquence f_i de chaque item pour le modèle eALS.
        """
        total_interactions = df.count()
        
        item_counts = df.groupBy("item_idx").agg(F.count("*").alias("count"))
        df_popularity = item_counts.withColumn("f_i", F.col("count") / F.lit(total_interactions))
        
        df_popularity = df_popularity.select("item_idx", "f_i").cache()
        df_popularity.count() 
        
        return df_popularity

    def transform(self, df_raw: DataFrame, min_interactions: int = 10) -> Tuple[DataFrame, DataFrame]:
        """
        Exécute la pipeline complète de transformation.
        """
        df_base = df_raw.select(self.user_col, self.item_col)
        df_filtered = self._filter_interactions(df_base, min_interactions)
        df_indexed = self._create_integer_indices(df_filtered)
        df_popularity = self._compute_item_popularity(df_indexed)

        df_final = df_indexed.withColumn("rating", F.lit(1.0).cast("float")) \
                             .select("user_idx", "item_idx", "rating")
        
        df_final = df_final.repartition("user_idx").cache()
        df_final.count() 

        print("--- Pipeline de données terminée avec succès ! ---")
        return df_final, df_popularity
    
import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def run_test():
    # 1. Initialisation de la SparkSession en mode local
    print("Initialisation de Spark...")
    spark = SparkSession.builder \
        .appName("Test_eALS_Preprocessor") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    #  Définir un dossier pour les localCheckpoints
    checkpoint_dir = "data/spark_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    # 2. Création d'un jeu de données factice (Mock Data)
    # On va simuler des utilisateurs et des items. 
    # Pour ce test, on mettra un seuil min_interactions = 3
    print("Création du dataset de test...")
    data = [
        # User 1 est très actif (3 interactions)
        ("user_1", "item_A"), ("user_1", "item_B"), ("user_1", "item_C"),
        # User 2 est très actif (3 interactions)
        ("user_2", "item_A"), ("user_2", "item_B"), ("user_2", "item_D"),
        # User 3 est inactif (1 interaction) -> DEVRAIT ÊTRE SUPPRIMÉ
        ("user_3", "item_A"),
        
        # item_A a 3 interactions (valide)
        # item_B a 2 interactions (invalide, DEVRAIT ÊTRE SUPPRIMÉ, 
        #   ce qui fera chuter User 1 et User 2 sous le seuil des 3 interactions s'ils ne sont pas assez solides !)
        # item_C a 1 interaction (invalide)
        # item_D a 1 interaction (invalide)
        
        # Ajoutons un peu de volume pour stabiliser user 1 et 2 et l'item A et E
        ("user_1", "item_E"), ("user_2", "item_E"), ("user_4", "item_E"),
        ("user_4", "item_A"), ("user_4", "item_F"), ("user_4", "item_G")
    ]
    
    # Pour simplifier le test mental, disons min_interactions = 2
    # - user_3 saute (1 int)
    # - item_C et item_D sautent (1 int)
    # Ce qui va affecter user_1 et user_2, etc. La boucle while fera son travail !
    
    df_raw = spark.createDataFrame(data, ["user_id", "business_id"])
    print("\n--- DataFrame Brut ---")
    df_raw.show()

    # 3. Test de la classe ImplicitDataPreprocessor
    preprocessor = ImplicitDataPreprocessor(user_col="user_id", item_col="business_id")
    
    # On utilise un seuil de 2 pour ce mini-test
    df_interactions, df_popularity = preprocessor.transform(df_raw, min_interactions=2)

    # 4. Affichage des résultats finaux
    print("\n--- Matrice d'Interactions Finale (Indexée) ---")
    df_interactions.show()
    
    print("\n--- Schéma de la Matrice d'Interactions ---")
    df_interactions.printSchema()

    print("\n--- Popularité des Items (f_i) ---")
    df_popularity.show()

    # Optionnel : Tester avec le VRAI dataset Yelp
    # Pour tester avec Yelp, décommente les lignes ci-dessous :
    """
    print("\n--- Test sur Yelp (Extrait) ---")
    df_yelp = spark.read.json("chemin/vers/yelp_academic_dataset_review.json")
    df_yelp_interactions, df_yelp_pop = preprocessor.transform(df_yelp, min_interactions=10)
    df_yelp_interactions.show(5)
    """

    spark.stop()

if __name__ == "__main__":
    run_test()
