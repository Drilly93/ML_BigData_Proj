import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from data_prep import ImplicitDataPreprocessor
from eALS_pyspark import PySpark_eALS




def run_test():
    # 1. Initialisation de la SparkSession en mode local
    print("Initialisation de Spark...")
    spark = SparkSession.builder \
        .appName("Test_eALS_Preprocessor") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # CRITIQUE : Définir un dossier pour les localCheckpoints
    # Sur Windows, utilise un chemin comme "C:/tmp/checkpoints"
    checkpoint_dir = "/tmp/spark_checkpoints"
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
# ... (fin de l'étape 1) ...
    
    # --- TEST DE L'ÉTAPE 2 ---
    print("\n--- ÉTAPE 2 : Modélisation eALS ---")
    # On initialise le modèle avec K=3 (pour que l'affichage console soit lisible)
    eals_model = PySpark_eALS(K=3, max_iter=5)
    
    # On génère les matrices P et Q
    matrice_P, matrice_Q = eals_model.init_latent_factors(df_interactions)
    
    print("\nAperçu de la Matrice P (Utilisateurs) :")
    matrice_P.show(truncate=False)
    
    print("Aperçu de la Matrice Q (Items) :")
    matrice_Q.show(truncate=False)
    
    
    print("\n--- ÉTAPE 3 : Calcul des Caches ---")
    
    # 1. Calcul des confiances (à faire une fois avant la boucle)
    df_c_i = eals_model._compute_item_confidences(df_popularity)
    print("Confiances (c_i) :")
    df_c_i.show()
    
    # 2. Mise à jour des caches (S^p et S^q)
    eals_model._update_caches(df_c_i, spark)
    
    # 3. Vérification
    Sq_matrix = eals_model.broadcast_Sq.value
    Sp_matrix = eals_model.broadcast_Sp.value
    
    print(f"Shape de S^q : {Sq_matrix.shape}") # Devrait afficher (3, 3) car K=3
    print(f"Shape de S^p : {Sp_matrix.shape}")

if __name__ == "__main__":
    run_test()