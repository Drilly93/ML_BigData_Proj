import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from typing import Tuple
from pyspark.broadcast import Broadcast
import pyspark.sql.types as T



class PySpark_eALS:
    def __init__(self, K: int = 128, lambda_reg: float = 0.01, c0: float = 512.0, alpha: float = 0.4, max_iter: int = 10):
        """
        Args:
            K (int): Dimensionnalité des facteurs latents.
            lambda_reg (float): Régularisation L2 pour prévenir le surapprentissage[cite: 106, 109].
            c0 (float): Poids global pour les données manquantes (feedback négatif)[cite: 180].
            alpha (float): Exposant pour pondérer les items selon leur popularité[cite: 181].
            max_iter (int): Nombre maximum d'itérations pour l'entraînement.
        
        Valeurs par défaut (c0=512, alpha=0.4, K=128)
        """
        self.K = K
        self.lambda_reg = lambda_reg
        self.c0 = c0
        self.alpha = alpha
        self.max_iter = max_iter

        self.P: DataFrame = None  # Matrice des utilisateurs
        self.Q: DataFrame = None  # Matrice des items
        self.broadcast_Sq = None
        self.broadcast_Sp = None

    def init_latent_factors(self, df_interactions: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Initialisation de  P et Q
        
        Args:
            df_interactions (DataFrame): Matrice contenant au moins "user_idx" et "item_idx".
        """
        
        random_vector_expr = F.array([F.rand() * 0.01 for _ in range(self.K)]) # Vecteur aléatoire

        # Initialisation de P
        df_users = df_interactions.select("user_idx").distinct()
        self.P = df_users.withColumn("factors", random_vector_expr)
        self.P = self.P.repartition("user_idx").cache()
        num_users = self.P.count() # Force la matérialisation du cache

        # Initialisation de Q
        df_items = df_interactions.select("item_idx").distinct()
        self.Q = df_items.withColumn("factors", random_vector_expr)
        self.Q = self.Q.repartition("item_idx").cache()
        num_items = self.Q.count()

        return self.P, self.Q
    
    
    def _compute_item_confidences(self, df_popularity: DataFrame) -> DataFrame:
        """
        Calcule le score de popularité c_i de chaque item
        """
        
        # Calcul de c_i
        df_pop_alpha = df_popularity.withColumn("f_i_alpha", F.pow("f_i", self.alpha))
        sum_f_alpha = df_pop_alpha.select(F.sum("f_i_alpha")).collect()[0][0]
        
        df_c_i = df_pop_alpha.withColumn(
            "c_i", 
            (F.col("f_i_alpha") / F.lit(sum_f_alpha)) * F.lit(self.c0)
        ).select("item_idx", "c_i")
        
        df_c_i.cache() # DataFrame en cache, car il sera utilisé à CHAQUE itération
        df_c_i.count()
        
        return df_c_i
    

    def _update_caches(self, df_c_i: DataFrame, spark_session):
        K = self.K
        q_with_c = self.Q.join(F.broadcast(df_c_i), on="item_idx", how="inner")
        
        # Calcul Sq avec mapPartitions pour éviter les shuffles massifs
        def compute_Sq_partition(iterator):
            import numpy as np
            local_Sq = np.zeros((K, K), dtype=np.float32)
            for row in iterator:
                q_vec = np.array(row['factors'], dtype=np.float32)
                c_i = row['c_i']
                local_Sq += c_i * np.outer(q_vec, q_vec)
                
            yield local_Sq
        Sq_matrices = q_with_c.rdd.mapPartitions(compute_Sq_partition).collect()
        Sq_final = sum(Sq_matrices) if Sq_matrices else np.zeros((K, K), dtype=np.float32)

        # Calcul Sp
        def compute_Sp_partition(iterator):
            import numpy as np
            local_Sp = np.zeros((K, K), dtype=np.float32)
            for row in iterator:
                p_vec = np.array(row['factors'], dtype=np.float32)
                local_Sp += np.outer(p_vec, p_vec)
            yield local_Sp
        Sp_matrices = self.P.rdd.mapPartitions(compute_Sp_partition).collect()
        Sp_final = sum(Sp_matrices) if Sp_matrices else np.zeros((K, K), dtype=np.float32)

        # Création de Broadcast
        if self.broadcast_Sq is not None:
            self.broadcast_Sq.unpersist()
        if self.broadcast_Sp is not None:
            self.broadcast_Sp.unpersist()
            
        self.broadcast_Sq = spark_session.sparkContext.broadcast(Sq_final)
        self.broadcast_Sp = spark_session.sparkContext.broadcast(Sp_final)
        
        print("-> Caches calculés à la vitesse NumPy et diffusés.")


    def _update_P(self, df_interactions: DataFrame, df_c_i: DataFrame, spark) -> DataFrame:
        """
        Met à jour la matrice P
        """
        
        # Dataset avec (q_i, c_i, rating) pour chaque interaction utilisateur-item
        df_joined = df_interactions.join(self.Q, on="item_idx", how="inner") \
                                   .join(F.broadcast(df_c_i), on="item_idx", how="inner")
        
        # Conversion en une liste
        df_grouped = df_joined.groupBy("user_idx").agg(
            F.collect_list(F.struct("rating", "factors", "c_i")).alias("interactions")
        )

        # Variables
        K = self.K
        lambda_reg = self.lambda_reg
        Sq_bc = self.broadcast_Sq

        # Exécuté localement sur chaque cœur CPU du cluster
        def process_user(row):
            import numpy as np # Import nécessaire sur les workers
            user_idx = row.user_idx
            interactions = row.interactions
            Sq = Sq_bc.value
            
            p_u = np.zeros(K, dtype=np.float32)
            num_interactions = len(interactions)
            r_hat = np.zeros(num_interactions, dtype=np.float32) # Prédiction en cours
            
            if num_interactions > 0:
                Q_mat = np.array([inter.factors for inter in interactions], dtype=np.float32)
                ratings = np.array([inter.rating for inter in interactions], dtype=np.float32)
                weights = 1.0 - np.array([inter.c_i for inter in interactions], dtype=np.float32)
            else:
                return (user_idx, p_u.tolist())

            for f in range(K):
                old_val = p_u[f]
                
                numerator = -(np.dot(p_u, Sq[:, f]) - old_val * Sq[f, f])
                denominator = Sq[f, f] + lambda_reg
                
                if num_interactions > 0:
                    q_f = Q_mat[:, f] # Colonne de la dimension f pour tous les items de l'user
                    r_hat_f = r_hat - old_val * q_f
                    numerator += np.sum((ratings - weights * r_hat_f) * q_f)
                    denominator += np.sum(weights * (q_f ** 2))
                
                p_u[f] = numerator / denominator
                
                # Prédiction
                if num_interactions > 0:
                    r_hat = r_hat_f + p_u[f] * q_f
                    
            return (user_idx, p_u.tolist())

        # RETOUR À SPARK
        schema = T.StructType([
            T.StructField("user_idx", T.IntegerType(), False),
            T.StructField("factors", T.ArrayType(T.FloatType()), False)
        ])
        
        new_P = spark.createDataFrame(df_grouped.rdd.map(process_user), schema)
        return new_P.repartition("user_idx")

    def _update_Q(self, df_interactions: DataFrame, df_c_i: DataFrame, spark) -> DataFrame:
        """
        Met à jour la matrice Q
        """
        
        # Jointure avec P (Utilisateurs)
        df_joined = df_interactions.join(self.P, on="user_idx", how="inner")
        
        df_grouped = df_joined.groupBy("item_idx").agg(
            F.collect_list(F.struct("rating", "factors")).alias("interactions")
        )
        # Rajout de c_i
        df_grouped = df_grouped.join(F.broadcast(df_c_i), on="item_idx", how="inner")

        K = self.K
        lambda_reg = self.lambda_reg
        Sp_bc = self.broadcast_Sp

        # Exécuté localement sur chaque cœur CPU du cluster
        def process_item(row):
            import numpy as np
            item_idx = row.item_idx
            c_i = row.c_i
            interactions = row.interactions
            Sp = Sp_bc.value
            
            q_i = np.zeros(K, dtype=np.float32)
            num_interactions = len(interactions)
            r_hat = np.zeros(num_interactions, dtype=np.float32)
            
            if num_interactions > 0:
                P_mat = np.array([inter.factors for inter in interactions], dtype=np.float32)
                ratings = np.array([inter.rating for inter in interactions], dtype=np.float32)
                weights = 1.0 - c_i 
            else:
                return (item_idx, q_i.tolist())

            for f in range(K):
                old_val = q_i[f]
                numerator = -c_i * (np.dot(q_i, Sp[:, f]) - old_val * Sp[f, f])
                denominator = c_i * Sp[f, f] + lambda_reg
                
                if num_interactions > 0:
                    p_f = P_mat[:, f]
                    r_hat_f = r_hat - old_val * p_f
                    numerator += np.sum((ratings - weights * r_hat_f) * p_f)
                    denominator += np.sum(weights * (p_f ** 2))
                
                q_i[f] = numerator / denominator
                
                # Prédiction
                if num_interactions > 0:
                    r_hat = r_hat_f + q_i[f] * p_f
                    
            return (item_idx, q_i.tolist())

        schema = T.StructType([
            T.StructField("item_idx", T.IntegerType(), False),
            T.StructField("factors", T.ArrayType(T.FloatType()), False)
        ])
        
        new_Q = spark.createDataFrame(df_grouped.rdd.map(process_item), schema)
        return new_Q.repartition("item_idx")

    def fit(self, df_interactions: DataFrame, df_popularity: DataFrame, spark):
        """
        Entraîne le modèle eALS en utilisant l'alternance (ALS)
        """
        print(f"\n====== DÉBUT DE L'ENTRAÎNEMENT eALS ({self.max_iter} Itérations) ======")
        
        # Initialisation
        df_c_i = self._compute_item_confidences(df_popularity)
        self.init_latent_factors(df_interactions)
        
        for iteration in range(1, self.max_iter + 1):
            print(f"\n--- [Itération {iteration}/{self.max_iter}] ---")
    
            self._update_caches(df_c_i, spark) # Met à jour S^q et S^p

            # Update P
            new_P = self._update_P(df_interactions, df_c_i, spark)
            new_P = new_P.localCheckpoint()
            new_P.count() # Force l'évaluation
            self.P.unpersist() # Libère l'ancienne matrice de la RAM
            self.P = new_P

            # Update Q
            self._update_caches(df_c_i, spark) # Recalcul requis car P vient de changer !
            new_Q = self._update_Q(df_interactions, df_c_i, spark)
            new_Q = new_Q.localCheckpoint()
            new_Q.count()
            self.Q.unpersist()
            self.Q = new_Q
            
            print(f"Itération {iteration} complétée avec succès.")

        print("====== ENTRAÎNEMENT TERMINÉ ======")



    def evaluate_model_fast(self, df_test: DataFrame, spark) -> tuple[float, float]:
        """
        Évalue le Hit Ratio (HR@10) et le NDCG@10 via la méthode du Negative Sampling (1 vs 99).
        """
        import pandas as pd
        import numpy as np
        from pyspark.sql.window import Window
        
        #ÉCHANTILLONNAGE (Sur le Driver via Pandas car 500 lignes = instantané)
        # On rapatrie le petit set de test localement
        test_pd = df_test.select("user_idx", "item_idx").toPandas()
        total_items = self.Q.count()
        
        eval_data = []
        for _, row in test_pd.iterrows():
            u = int(row['user_idx'])
            true_i = int(row['item_idx'])
            
            # Le vrai item (is_true = 1)
            eval_data.append((u, true_i, 1))
            
            # 99 Faux items tirés au hasard (is_true = 0)
            negatives = np.random.randint(0, total_items, 99)
            for neg_i in negatives:
                eval_data.append((u, int(neg_i), 0))
                
        # On renvoie les 50 000 couples (500 users * 100 items) dans Spark
        df_eval = spark.createDataFrame(eval_data, ["user_idx", "item_idx", "is_true"])
        
        # JOINTURE BROADCAST (Extrêmement rapide car df_eval est petit)
        df_scored = df_eval.join(F.broadcast(self.P.withColumnRenamed("factors", "p_factors")), on="user_idx") \
                           .join(F.broadcast(self.Q.withColumnRenamed("factors", "q_factors")), on="item_idx")
                           
        # LE TUNGSTEN SQL TRICK (Produit scalaire natif sans UDF)
        # On génère la requête : "p_factors[0]*q_factors[0] + p_factors[1]*q_factors[1]..."
        dot_expr = " + ".join([f"(p_factors[{i}] * q_factors[{i}])" for i in range(self.K)])
        df_scored = df_scored.withColumn("score", F.expr(dot_expr))
        
        # CLASSEMENT (RANKING)
        # On classe les 100 items de chaque utilisateur du plus grand score au plus petit
        window_spec = Window.partitionBy("user_idx").orderBy(F.col("score").desc())
        df_ranked = df_scored.withColumn("rank", F.row_number().over(window_spec))
        
        # CALCUL DES MÉTRIQUES (Sur le vrai item uniquement)
        df_results = df_ranked.filter(F.col("is_true") == 1)
        
        # Formules académiques pour @10 : 
        # HR = 1 si rank <= 10, sinon 0
        # NDCG = ln(2) / ln(rank+1) si rank <= 10, sinon 0
        df_metrics = df_results.withColumn("hit", F.when(F.col("rank") <= 10, 1.0).otherwise(0.0)) \
                               .withColumn("ndcg", F.when(F.col("rank") <= 10, F.lit(0.693147) / F.log(F.col("rank") + 1)).otherwise(0.0))
                               
        metrics = df_metrics.agg(F.mean("hit").alias("hr"), F.mean("ndcg").alias("ndcg")).collect()[0]
        
        return metrics["hr"], metrics["ndcg"]


    def fit_with_telemetry(self, df_train: DataFrame, df_test: DataFrame, df_popularity: DataFrame, spark):
        """
        Entraîne le modèle avec un tableau de bord complet (Temps, HR, Opérations, Réseau).
        Remplace l'ancienne méthode `fit`.
        """
        import time
        print(f"\n{'='*60}")
        print(f" DÉMARRAGE MOTEUR eALS - {self.max_iter} ITÉRATIONS (K={self.K})")
        print(f"{'='*60}")
        
        # Statistiques initiales
        M = df_train.select("user_idx").distinct().count()
        
        # Initialisation
        df_c_i = self._compute_item_confidences(df_popularity)
        self.init_latent_factors(df_train)
        N = self.Q.count()
        
        print(f"\n[INFO] Dataset : {M} Utilisateurs | {N} Items")
        print("-" * 60)

        for iteration in range(1, self.max_iter + 1):
            start_time = time.time()
            
            # Update P
            self._update_caches(df_c_i, spark)
            new_P = self._update_P(df_train, df_c_i, spark).localCheckpoint()
            new_P.count()
            self.P.unpersist()
            self.P = new_P

            # Update Q
            self._update_caches(df_c_i, spark)
            new_Q = self._update_Q(df_train, df_c_i, spark).localCheckpoint()
            new_Q.count()
            self.Q.unpersist()
            self.Q = new_Q
            
            # Indicateurs de performance à la fin de chaque itération

            # --- Télémétrie : Temps ---
            elapsed_time = round(time.time() - start_time, 2)
            
            # --- Télémétrie : Évaluation ---
            hr_10, ndcg_10 = self.evaluate_model_fast(df_test, spark)
            
            # --- Télémétrie : Complexité ---
            # Unité : millions d'opérations (Mops)
            ops_classique = ((M + N) * (self.K ** 3)) / 1_000_000
            ops_eALS = ((M + N) * (self.K ** 2)) / 1_000_000
            facteur_gain = self.K
            
            # Affichage du Dashboard
            print(f"\n[ ITERATION {iteration}/{self.max_iter} TERMINEE en {elapsed_time}s ]")
            print(f" Qualité   | HR@10: {hr_10:.4f}  | NDCG@10: {ndcg_10:.4f}")
            print(f" Calculs   | eALS: {ops_eALS:.1f} Mops | Classique: {ops_classique:.1f} Mops")

            print("-" * 60)
            
        print("\n====== ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ======")