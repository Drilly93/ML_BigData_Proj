import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from typing import Tuple
from pyspark.broadcast import Broadcast

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
        broadcast_Sq: Broadcast = None
        broadcast_Sp: Broadcast = None
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