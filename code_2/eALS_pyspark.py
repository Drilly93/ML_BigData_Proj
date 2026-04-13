import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from typing import Tuple
from pyspark.broadcast import Broadcast
import pyspark.sql.types as T



class PySpark_eALS:
    """
    Implémentation distribuée et optimisée de l'algorithme eALS (element-wise Alternating Least Squares)
    pour la recommandation avec feedback implicite.
    """

    def __init__(self, K: int = 128, lambda_reg: float = 0.01, c0: float = 512.0, alpha: float = 0.4, max_iter: int = 10):
        """
        Initialise les hyperparamètres du modèle eALS. 
        Les valeurs par défaut (c0=512, alpha=0.4, K=128) correspondent aux paramètres 
        optimaux trouvés pour le dataset Yelp dans l'article de recherche d'origine[cite: 488, 526].

        Args:
            K (int): Dimensionnalité des facteurs latents.
            lambda_reg (float): Régularisation L2 pour prévenir le surapprentissage[cite: 106, 109].
            c0 (float): Poids global pour les données manquantes (feedback négatif)[cite: 180].
            alpha (float): Exposant pour pondérer les items selon leur popularité[cite: 181].
            max_iter (int): Nombre maximum d'itérations pour l'entraînement.
        """
        self.K = K
        self.lambda_reg = lambda_reg
        self.c0 = c0
        self.alpha = alpha
        self.max_iter = max_iter

        # Réservation de l'espace pour les futures matrices latentes distribuées
        self.P: DataFrame = None  # Matrice des utilisateurs
        self.Q: DataFrame = None  # Matrice des items
        self.broadcast_Sq = None
        self.broadcast_Sp = None
    def init_latent_factors(self, df_interactions: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Génère les matrices P et Q initiales à partir de la matrice d'interactions.
        Les facteurs sont initialisés avec de très petites valeurs aléatoires pour 
        casser la symétrie sans faire exploser les gradients.

        Args:
            df_interactions (DataFrame): Matrice contenant au moins "user_idx" et "item_idx".

        Returns:
            Tuple[DataFrame, DataFrame]: Les DataFrames P et Q mis en cache et partitionnés.
        """
        print(f"--- Initialisation des matrices P et Q (K={self.K}) ---")

        # OPTIMISATION MAJEURE : Création d'une expression SQL native.
        # Plutôt que d'utiliser une UDF Python lente, on demande au moteur Spark (Tungsten)
        # de créer un tableau (Array) de K valeurs aléatoires (multipliées par 0.01 pour rester petites).
        random_vector_expr = F.array([F.rand() * 0.01 for _ in range(self.K)])

        # -------------------------------------------------------------------
        # 1. Initialisation de la Matrice P (Utilisateurs)
        # -------------------------------------------------------------------
        print("Génération de P...")
        # On extrait la liste exhaustive des utilisateurs
        df_users = df_interactions.select("user_idx").distinct()
        
        # On ajoute le vecteur latent généré nativement
        self.P = df_users.withColumn("factors", random_vector_expr)
        
        # OPTIMISATION RÉSEAU : Le repartitionnement par "user_idx" garantit que 
        # toutes les données d'un même utilisateur vivront sur le même processeur,
        # évitant ainsi les "shuffles" destructeurs de performances lors de l'entraînement.
        self.P = self.P.repartition("user_idx").cache()
        num_users = self.P.count() # Force la matérialisation du cache
        print(f"-> Matrice P initialisée et mise en cache pour {num_users} utilisateurs.")

        # -------------------------------------------------------------------
        # 2. Initialisation de la Matrice Q (Items)
        # -------------------------------------------------------------------
        print("Génération de Q...")
        df_items = df_interactions.select("item_idx").distinct()
        
        self.Q = df_items.withColumn("factors", random_vector_expr)
        
        self.Q = self.Q.repartition("item_idx").cache()
        num_items = self.Q.count() # Force la matérialisation du cache
        print(f"-> Matrice Q initialisée et mise en cache pour {num_items} items.")

        return self.P, self.Q
    
    
    def _compute_item_confidences(self, df_popularity: DataFrame) -> DataFrame:
        """
        Étape 3A : Calcule la confiance c_i pour chaque item basée sur sa popularité.
        Formule : c_i = c0 * (f_i^alpha / sum(f_j^alpha))
        """
        print("--- Précalcul des poids de confiance c_i ---")
        
        # 1. Calculer f_i^alpha
        df_pop_alpha = df_popularity.withColumn("f_i_alpha", F.pow("f_i", self.alpha))
        
        # 2. Récupérer la somme totale (action scalaire qui remonte au Driver)
        sum_f_alpha = df_pop_alpha.select(F.sum("f_i_alpha")).collect()[0][0]
        
        # 3. Calculer le c_i final
        df_c_i = df_pop_alpha.withColumn(
            "c_i", 
            (F.col("f_i_alpha") / F.lit(sum_f_alpha)) * F.lit(self.c0)
        ).select("item_idx", "c_i")
        
        # On met ce petit DataFrame en cache, car il sera utilisé à CHAQUE itération
        df_c_i.cache()
        df_c_i.count() # Force la matérialisation
        
        return df_c_i

    def _update_caches(self, df_c_i: DataFrame, spark_session):
        """
        Étape 3B : Calcule les matrices globales de cache S^q et S^p via 
        une agrégation distribuée massivement parallèle (treeAggregate + numpy).
        """
        print("--- Mise à jour des Caches Globaux S^q et S^p ---")
        
        K = self.K
        
        # ---------------------------------------------------------
        # Calcul de S^q = sum(c_i * q_i * q_i^T)
        # ---------------------------------------------------------
        # Optimisation : Broadcast Join car df_c_i est très petit (N items, 2 colonnes)
        q_with_c = self.Q.join(F.broadcast(df_c_i), on="item_idx", how="inner")
        
        # Définition des fonctions pour l'agrégation RDD
        def seq_op_Sq(acc: np.ndarray, row) -> np.ndarray:
            q_vec = np.array(row['factors'], dtype=np.float32)
            c_i = row['c_i']
            # Ajoute le produit externe pondéré à l'accumulateur local de la partition
            return acc + c_i * np.outer(q_vec, q_vec)
            
        def comb_op(acc1: np.ndarray, acc2: np.ndarray) -> np.ndarray:
            # Combine les accumulateurs des différentes partitions
            return acc1 + acc2

        # treeAggregate est magique : il fait le reduce localement puis hiérarchiquement
        # evitant un crash OOM sur le Driver si on a beaucoup de partitions.
        Sq_local = q_with_c.rdd.treeAggregate(
            zeroValue=np.zeros((K, K), dtype=np.float32),
            seqOp=seq_op_Sq,
            combOp=comb_op,
            depth=3
        )

        # ---------------------------------------------------------
        # Calcul de S^p = P^T * P = sum(p_u * p_u^T)
        # ---------------------------------------------------------
        def seq_op_Sp(acc: np.ndarray, row) -> np.ndarray:
            p_vec = np.array(row['factors'], dtype=np.float32)
            return acc + np.outer(p_vec, p_vec)

        Sp_local = self.P.rdd.treeAggregate(
            zeroValue=np.zeros((K, K), dtype=np.float32),
            seqOp=seq_op_Sp,
            combOp=comb_op,
            depth=3
        )

        # ---------------------------------------------------------
        # Libération des anciens broadcasts et création des nouveaux
        # ---------------------------------------------------------
        if self.broadcast_Sq is not None:
            self.broadcast_Sq.unpersist()
        if self.broadcast_Sp is not None:
            self.broadcast_Sp.unpersist()
            
        self.broadcast_Sq = spark_session.sparkContext.broadcast(Sq_local)
        self.broadcast_Sp = spark_session.sparkContext.broadcast(Sp_local)
        
        print("-> Caches S^q et S^p calculés et diffusés avec succès.")


    def _update_P(self, df_interactions: DataFrame, df_c_i: DataFrame, spark) -> DataFrame:
        """
        Met à jour la matrice latente des Utilisateurs (P) élément par élément.
        """
        print("-> Mise à jour de P (Utilisateurs)...")
        
        # 1. PRÉPARATION SPARK SQL (Ultra rapide pour les jointures)
        # On regroupe toutes les infos (q_i, c_i, rating) nécessaires pour chaque utilisateur
        df_joined = df_interactions.join(self.Q, on="item_idx", how="inner") \
                                   .join(F.broadcast(df_c_i), on="item_idx", how="inner")
        
        # On crée une liste de "Structs" par utilisateur pour traiter tout en une fois
        df_grouped = df_joined.groupBy("user_idx").agg(
            F.collect_list(F.struct("rating", "factors", "c_i")).alias("interactions")
        )

        # Extraction des hyperparamètres pour le RDD
        K = self.K
        lambda_reg = self.lambda_reg
        Sq_bc = self.broadcast_Sq

        # 2. CALCUL NUMPY (Exécuté localement sur chaque cœur CPU du cluster)
        def process_user(row):
            import numpy as np # Import nécessaire sur les workers
            user_idx = row.user_idx
            interactions = row.interactions
            Sq = Sq_bc.value
            
            p_u = np.zeros(K, dtype=np.float32)
            num_interactions = len(interactions)
            r_hat = np.zeros(num_interactions, dtype=np.float32) # Prédiction en cours
            
            if num_interactions > 0:
                # OPTIMISATION EXTRÊME : Vectorisation NumPy
                # Au lieu de faire des boucles Python sur les interactions, on crée des matrices
                Q_mat = np.array([inter.factors for inter in interactions], dtype=np.float32)
                ratings = np.array([inter.rating for inter in interactions], dtype=np.float32)
                weights = 1.0 - np.array([inter.c_i for inter in interactions], dtype=np.float32)
            else:
                return (user_idx, p_u.tolist())

            # Boucle eALS : Mise à jour dimension par dimension
            for f in range(K):
                old_val = p_u[f]
                
                # Partie 1 : Espace négatif (via le cache global Sq)
                # Astuce mathématique : On soustrait l'ancienne valeur pour ne pas s'inclure soi-même
                numerator = -(np.dot(p_u, Sq[:, f]) - old_val * Sq[f, f])
                denominator = Sq[f, f] + lambda_reg
                
                # Partie 2 : Espace positif (les interactions observées)
                if num_interactions > 0:
                    q_f = Q_mat[:, f] # Colonne de la dimension f pour tous les items de l'user
                    
                    # On retire l'influence de la dimension actuelle de la prédiction globale
                    r_hat_f = r_hat - old_val * q_f
                    
                    # On ajoute l'influence des vraies interactions
                    numerator += np.sum((ratings - weights * r_hat_f) * q_f)
                    denominator += np.sum(weights * (q_f ** 2))
                
                # Mise à jour exacte
                p_u[f] = numerator / denominator
                
                # On met à jour la prédiction pour la prochaine dimension
                if num_interactions > 0:
                    r_hat = r_hat_f + p_u[f] * q_f
                    
            return (user_idx, p_u.tolist())

        # 3. RETOUR À SPARK
        schema = T.StructType([
            T.StructField("user_idx", T.IntegerType(), False),
            T.StructField("factors", T.ArrayType(T.FloatType()), False)
        ])
        
        new_P = spark.createDataFrame(df_grouped.rdd.map(process_user), schema)
        return new_P.repartition("user_idx")

    def _update_Q(self, df_interactions: DataFrame, df_c_i: DataFrame, spark) -> DataFrame:
        """
        Met à jour la matrice latente des Items (Q) élément par élément.
        Symétrique à P, mais attention : la confiance c_i est fixe pour un item !
        """
        print("-> Mise à jour de Q (Items)...")
        
        # Jointure avec P (Utilisateurs)
        df_joined = df_interactions.join(self.P, on="user_idx", how="inner")
        
        df_grouped = df_joined.groupBy("item_idx").agg(
            F.collect_list(F.struct("rating", "factors")).alias("interactions")
        )
        # On rajoute c_i à l'item
        df_grouped = df_grouped.join(F.broadcast(df_c_i), on="item_idx", how="inner")

        K = self.K
        lambda_reg = self.lambda_reg
        Sp_bc = self.broadcast_Sp

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
                # Attention : Pour l'item, le cache Sp est multiplié par c_i (Équation 13)
                numerator = -c_i * (np.dot(q_i, Sp[:, f]) - old_val * Sp[f, f])
                denominator = c_i * Sp[f, f] + lambda_reg
                
                if num_interactions > 0:
                    p_f = P_mat[:, f]
                    r_hat_f = r_hat - old_val * p_f
                    numerator += np.sum((ratings - weights * r_hat_f) * p_f)
                    denominator += np.sum(weights * (p_f ** 2))
                
                q_i[f] = numerator / denominator
                
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
        Entraîne le modèle eALS en utilisant l'alternance (ALS) avec gestion du Checkpointing.
        """
        print(f"\n====== DÉBUT DE L'ENTRAÎNEMENT eALS ({self.max_iter} Itérations) ======")
        
        # 1. Initialisation des poids c_i et des matrices P et Q
        df_c_i = self._compute_item_confidences(df_popularity)
        self.init_latent_factors(df_interactions)
        
        # 2. Boucle principale Alternating Least Squares
        for iteration in range(1, self.max_iter + 1):
            print(f"\n--- [Itération {iteration}/{self.max_iter}] ---")
            
            # --- UPDATE UTILISATEURS ---
            self._update_caches(df_c_i, spark) # Met à jour S^q et S^p
            new_P = self._update_P(df_interactions, df_c_i, spark)
            
            # CHECKPOINTING VITAL : Coupe le graphe d'exécution pour éviter les MemoryError
            new_P = new_P.localCheckpoint()
            new_P.count() # Force l'évaluation
            
            # Libère l'ancienne matrice de la RAM
            self.P.unpersist()
            self.P = new_P

            # --- UPDATE ITEMS ---
            self._update_caches(df_c_i, spark) # Recalcul requis car P vient de changer !
            new_Q = self._update_Q(df_interactions, df_c_i, spark)
            
            new_Q = new_Q.localCheckpoint()
            new_Q.count()
            
            self.Q.unpersist()
            self.Q = new_Q
            
            print(f"Itération {iteration} complétée avec succès.")

        print("====== ENTRAÎNEMENT TERMINÉ ======")