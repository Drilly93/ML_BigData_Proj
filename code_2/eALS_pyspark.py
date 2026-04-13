from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from typing import Tuple, Dict

class PySpark_eALS:
    """
    Implémentation distribuée et optimisée de l'algorithme eALS (element-wise Alternating Least Squares)
    pour la recommandation avec feedback implicite, adaptée de (He et al., SIGIR 2016)[cite: 3, 6, 8, 17].
    """

    def __init__(self, K: int = 128, lambda_reg: float = 0.01, c0: float = 512.0, alpha: float = 0.4):
        """
        Initialise les hyperparamètres du modèle eALS[cite: 180, 181, 488].

        Args:
            K (int): Nombre de facteurs latents (dimensionnalité).
            lambda_reg (float): Paramètre de régularisation L2 pour éviter le surapprentissage[cite: 106, 109].
            c0 (float): Poids global accordé aux données manquantes (feedback négatif)[cite: 178, 180, 488].
            alpha (float): Exposant contrôlant la distribution du poids selon la popularité des items[cite: 178, 181, 488].
        """
        self.K = K
        self.lambda_reg = lambda_reg
        self.c0 = c0
        self.alpha = alpha
        
        # Modèles d'indexation pour pouvoir faire la correspondance inverse (ID Entier -> String)
        self.indexer_models: Dict[str, StringIndexerModel] = {}

    def _filter_interactions(self, df: DataFrame, min_interactions: int) -> DataFrame:
        """
        Filtre itérativement le DataFrame pour ne garder que les utilisateurs et les items 
        ayant au moins `min_interactions` interactions[cite: 344, 345, 346, 302].
        
        Cette méthode tourne en boucle jusqu'à ce que la taille du DataFrame se stabilise, 
        car la suppression d'un item peu populaire peut rendre un utilisateur inactif, et vice-versa.

        Args:
            df (DataFrame): Le DataFrame brut contenant "user_id" et "business_id".
            min_interactions (int): Le seuil minimum d'interactions requis.

        Returns:
            DataFrame: Le DataFrame filtré de son bruit (k-core filtering).
        """
        # Mise en cache initiale pour accélérer le premier comptage
        df.cache()
        current_count = df.count()
        previous_count = -1
        iteration = 1

        print(f"--- Début du filtrage (Seuil: {min_interactions}) ---")
        
        while current_count != previous_count:
            previous_count = current_count

            # 1. Filtrer les utilisateurs
            user_counts = df.groupBy("user_id").agg(F.count("*").alias("u_count"))
            valid_users = user_counts.filter(F.col("u_count") >= min_interactions).select("user_id")
            df = df.join(valid_users, on="user_id", how="inner")

            # 2. Filtrer les items
            item_counts = df.groupBy("business_id").agg(F.count("*").alias("i_count"))
            valid_items = item_counts.filter(F.col("i_count") >= min_interactions).select("business_id")
            df = df.join(valid_items, on="business_id", how="inner")

            # OPTIMISATION CRITIQUE : Couper le lineage Spark (DAG)
            # Sans localCheckpoint, la boucle while va créer un plan d'exécution infini 
            # et causer un dépassement de mémoire (StackOverflowError).
            df = df.localCheckpoint() 
            
            current_count = df.count()
            print(f"Itération {iteration}: {current_count} interactions restantes.")
            iteration += 1

        return df

    def _create_integer_indices(self, df: DataFrame) -> DataFrame:
        """
        Convertit les identifiants textuels (UUID) en entiers contigus allant de 0 à N-1.
        Ces entiers serviront d'indices pour accéder aux lignes des matrices latentes P et Q.

        Args:
            df (DataFrame): DataFrame contenant les colonnes string "user_id" et "business_id".

        Returns:
            DataFrame: DataFrame avec les nouvelles colonnes "user_idx" et "item_idx" (entiers).
        """
        # Indexation des utilisateurs
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx")
        user_model = user_indexer.fit(df)
        df = user_model.transform(df)
        
        # Indexation des items
        item_indexer = StringIndexer(inputCol="business_id", outputCol="item_idx")
        item_model = item_indexer.fit(df)
        df = item_model.transform(df)

        # Sauvegarde des modèles pour décoder les recommandations à la fin
        self.indexer_models['user'] = user_model
        self.indexer_models['item'] = item_model

        # Cast en entier natif pour optimiser les jointures ultérieures
        df = df.withColumn("user_idx", F.col("user_idx").cast("integer")) \
               .withColumn("item_idx", F.col("item_idx").cast("integer"))

        return df

    def _compute_item_popularity(self, df: DataFrame) -> DataFrame:
        """
        Calcule la popularité relative (fréquence f_i) de chaque item dans le dataset[cite: 178, 180].
        Formule : f_i = (Nombre d'interactions de l'item i) / (Nombre total d'interactions)[cite: 180].

        Args:
            df (DataFrame): DataFrame contenant la colonne "item_idx".

        Returns:
            DataFrame: DataFrame contenant "item_idx" et sa fréquence "f_i".
        """
        # Extraire le nombre total d'interactions sous forme scalaire (entier python)
        total_interactions = df.count()
        
        # Grouper par item pour compter les interactions absolues
        item_counts = df.groupBy("item_idx").agg(F.count("*").alias("count"))
        
        # Calculer la fréquence f_i
        df_popularity = item_counts.withColumn("f_i", F.col("count") / F.lit(total_interactions))
        
        # On garde uniquement l'index et la popularité
        df_popularity = df_popularity.select("item_idx", "f_i")
        
        # MISE EN CACHE : Ce petit DataFrame sera diffusé (broadcast) massivement plus tard
        df_popularity.cache()
        df_popularity.count() # Force la matérialisation du cache
        
        return df_popularity

    def prepare_data(self, df_raw: DataFrame, min_interactions: int = 10) -> Tuple[DataFrame, DataFrame]:
        """
        Orchestrateur de l'Étape 1 : Nettoyage, indexation et préparation des données 
        pour l'algorithme eALS[cite: 302, 344, 345].

        Args:
            df_raw (DataFrame): Le dataset brut (ex: yelp_academic_dataset_review).
            min_interactions (int, optionnel): Seuil pour le filtrage k-core[cite: 345]. Défaut à 10[cite: 302].

        Returns:
            Tuple[DataFrame, DataFrame]: 
                - df_final: Contient (user_idx, item_idx, rating=1.0)[cite: 110, 314].
                - df_popularity: Contient (item_idx, f_i)[cite: 178, 180].
        """
        print("1. Sélection des colonnes...")
        df_base = df_raw.select("user_id", "business_id")

        print("2. Filtrage des utilisateurs et items inactifs...")
        df_filtered = self._filter_interactions(df_base, min_interactions)

        print("3. Indexation des IDs en entiers...")
        df_indexed = self._create_integer_indices(df_filtered)

        print("4. Calcul de la popularité des items (f_i)...")
        df_popularity = self._compute_item_popularity(df_indexed)

        print("5. Ajout du signal implicite (rating = 1.0)...")
        # En recommandation implicite, chaque interaction observée vaut 1 [cite: 110, 314, 370]
        df_final = df_indexed.withColumn("rating", F.lit(1.0).cast("float")) \
                             .select("user_idx", "item_idx", "rating")
        
        # Ultime optimisation avant de passer à l'algorithme :
        # Repartitionner par user_idx pour préparer le terrain de l'ALS par élément
        df_final = df_final.repartition("user_idx")
        df_final.cache()
        df_final.count() # Force la matérialisation

        print("--- Préparation des données terminée ! ---")
        return df_final, df_popularity