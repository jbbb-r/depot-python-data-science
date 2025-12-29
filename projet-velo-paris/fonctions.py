import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv_file(fichier):
    """
    Charge un fichier CSV avec gestion automatique du format.
    """
    nom_fichier = fichier.split('/')[-1]
    print(f"\n{nom_fichier}")
    
    if '2023' in fichier:
        print("  Type : 2023 (NOUVEAU format avec header)")
        temp_df = pd.read_csv(
            fichier, 
            sep=',',
            engine='python',
            on_bad_lines='skip',
            dtype=str,
            encoding='utf-8'
        )
    else:
        print("  Type : Standard")
        try:
            temp_df = pd.read_csv(fichier, sep=';', on_bad_lines='skip', dtype=str)
            if temp_df.shape[1] < 5:
                temp_df = pd.read_csv(fichier, sep=',', on_bad_lines='skip', dtype=str)
        except:
            temp_df = pd.read_csv(fichier, sep=',', on_bad_lines='skip', dtype=str)
    
    print(f"  ✓ {len(temp_df):,} lignes chargées")
    return temp_df


def normalize_dataframe(df):
    """
    Normalise les types de colonnes d'un DataFrame.
    """
    print("\nNormalisation des types de données...")
    df.columns = df.columns.str.strip()
    
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    print("   Types normalisés")
    return df


def verify_years_coverage(df, col_date='Date et heure de comptage', sample_size=10000):
    """
    Vérifie la couverture temporelle des données.
    """
    print("\nVérification rapide des années présentes :")
    
    if col_date not in df.columns:
        print(f"⚠ Colonne '{col_date}' introuvable")
        return
    
    sample_dates = pd.to_datetime(
        df[col_date].sample(min(sample_size, len(df))), 
        format='ISO8601',
        utc=True,
        errors='coerce'
    )
    
    if sample_dates.notna().sum() > 0:
        annees = sample_dates.dt.year.value_counts().sort_index()
        print("\nAnnées détectées (échantillon) :")
        for annee, count in annees.items():
            print(f"  {annee}: ~{count:,} lignes dans l'échantillon")


def create_historical_parquet(data_dir, output_file):
    """
    Crée un fichier parquet consolidé à partir de tous les CSV.
    """
    print("="*80)
    print("CRÉATION DU PARQUET HISTORIQUE")
    print("="*80)
    
    fichiers = sorted(glob.glob(f"{data_dir}/*comptage*.csv"))
    print(f"\nFichiers trouvés : {len(fichiers)}")
    
    df_historique_list = []
    
    for fichier in fichiers:
        temp_df = load_csv_file(fichier)
        df_historique_list.append(temp_df)
    
    print("\n" + "="*80)
    print("CONSOLIDATION")
    print("="*80)
    
    df_historique = pd.concat(df_historique_list, ignore_index=True)
    print(f"\n Total combiné : {len(df_historique):,} lignes")
    
    df_historique = normalize_dataframe(df_historique)
    
    print("\nSauvegarde en parquet...")
    df_historique.to_parquet(output_file, compression='gzip')
    print(f"Sauvegardé : {output_file}")
    
    print("\n" + "="*80)
    print("RÉCAPITULATIF")
    print("="*80)
    print(f"Total de lignes : {len(df_historique):,}")
    print(f"Colonnes : {df_historique.columns.tolist()}")
    
    verify_years_coverage(df_historique)
    
    return df_historique


def load_or_create_parquet(data_dir, parquet_file):
    """
    Charge un fichier parquet existant ou le crée s'il n'existe pas.
    """
    if os.path.exists(parquet_file):
        print(f"Fichier : {parquet_file}")
        
        df_historique = pd.read_parquet(parquet_file)
        print(f"Lignes : {len(df_historique):,}")
        print(f"Colonnes : {list(df_historique.columns)}")
        print(f"   Pour recréer : import os; os.remove('{parquet_file}')")
        
        return df_historique
    else:
        return create_historical_parquet(data_dir, parquet_file)


def explore_data(df_historique):
    """
    Explore et affiche les statistiques du DataFrame.
    """
    print("="*80)
    print("EXPLORATION DES DONNÉES")
    print("="*80)
    
    print(f"\n INFORMATIONS GÉNÉRALES")
    print(f"{'='*80}")
    print(f"Nombre total de lignes : {len(df_historique):,}")
    print(f"Nombre de colonnes : {len(df_historique.columns)}")
    print(f"Taille en mémoire : {df_historique.memory_usage(deep=True).sum() / 1024**2:.2f} Mo")
    
    print(f"\n COLONNES")
    print(f"{'='*80}")
    for i, col in enumerate(df_historique.columns, 1):
        nb_non_null = df_historique[col].notna().sum()
        pct_non_null = nb_non_null / len(df_historique) * 100
        print(f"{i}. {col:<40} {nb_non_null:>12,} valeurs ({pct_non_null:>5.1f}%)")
    
    print(f"\n APERÇU DES DONNÉES ")
    print(f"{'='*80}")
    print(df_historique.head())
    
    print(f"\n STATISTIQUES SUR LES COMPTAGES")
    print(f"{'='*80}")
    comptages_num = pd.to_numeric(df_historique['Comptage horaire'], errors='coerce')
    print(f"Total de passages comptés : {comptages_num.sum():,.0f}")
    print(f"Moyenne par ligne : {comptages_num.mean():.2f}")
    print(f"Médiane : {comptages_num.median():.0f}")
    print(f"Maximum : {comptages_num.max():,.0f}")
    print(f"Minimum : {comptages_num.min():.0f}")
    
    print(f"\n COMPTEURS")
    print(f"{'='*80}")
    nb_compteurs = df_historique['Identifiant du point de comptage'].nunique()
    print(f"Nombre de compteurs uniques : {nb_compteurs:,}")
    
    print(f"\nTop 10 des sites les plus représentés :")
    top_sites = df_historique['Nom du point de comptage'].value_counts().head(10)
    for i, (site, count) in enumerate(top_sites.items(), 1):
        pct = count / len(df_historique) * 100
        print(f"  {i:2d}. {site:<50} {count:>10,} lignes ({pct:>5.2f}%)")
    
    print(f"\n PÉRIODE COUVERTE")
    print(f"{'='*80}")
    dates_sample = pd.to_datetime(
        df_historique['Date et heure de comptage'].sample(min(100000, len(df_historique))),
        format='ISO8601', 
        utc=True, 
        errors='coerce'
    )
    dates_valides = dates_sample.dropna()
    
    if len(dates_valides) > 0:
        print(f"Date la plus ancienne : {dates_valides.min()}")
        print(f"Date la plus récente : {dates_valides.max()}")
        print(f"Durée totale : {(dates_valides.max() - dates_valides.min()).days} jours")
        
        annees = dates_valides.dt.year.value_counts().sort_index()
        print(f"\nRépartition par année (échantillon de {len(dates_valides):,} lignes) :")
        for annee, count in annees.items():
            pct = count / len(dates_valides) * 100
            bar = '█' * int(pct / 2)
            print(f"  {int(annee)} : {bar:<50} {pct:>5.1f}%")
    
    print(f"\n COORDONNÉES GÉOGRAPHIQUES")
    print(f"{'='*80}")
    coords_non_null = df_historique['Coordonnées géographiques'].notna().sum()
    pct_coords = coords_non_null / len(df_historique) * 100
    print(f"Lignes avec coordonnées : {coords_non_null:,} ({pct_coords:.1f}%)")
    
    print(f"\nExemples de coordonnées :")
    coords_sample = df_historique['Coordonnées géographiques'].dropna().sample(min(3, coords_non_null))
    for coord in coords_sample:
        print(f"  {coord}")
    
    print(f"\n EXPLORATION TERMINÉE")
    print(f"{'='*80}\n")
    
    return df_historique


def analyze_bike_traffic(df_historique):
    """
    Analyse complète du trafic vélo : évolution mensuelle, par arrondissement et compteurs constants.
    """
    df = df_historique
    df.columns = df.columns.str.strip()
    
    col_mapping = {
        'Identifiant du point de comptage': 'id_compteur',
        'Nom du point de comptage': 'nom_site',
        'Comptage horaire': 'comptage',
        'Date et heure de comptage': 'date_heure',
        'Coordonnées géographiques': 'coordonnees',
    }
    df = df.rename(columns=col_mapping)
    
    print(f"Lignes de départ : {len(df):,}")
    
    print("\nConversion des dates...")
    df['date_heure_clean'] = pd.to_datetime(df['date_heure'], format='ISO8601', utc=True, errors='coerce')
    df['date_heure_clean'] = df['date_heure_clean'].dt.tz_localize(None)
    
    nb_converti = df['date_heure_clean'].notna().sum()
    print(f"   Converti : {nb_converti:,} / {len(df):,}")
    
    df = df.dropna(subset=['date_heure_clean'])
    
    df['annee_mois'] = df['date_heure_clean'].dt.to_period('M')
    df['annee'] = df['date_heure_clean'].dt.year
    
    print(f"\nRépartition par année :")
    for annee, count in df['annee'].value_counts().sort_index().items():
        print(f"  {annee}: {count:,} lignes")
    
    print("\nConversion des comptages...")
    df['comptage_num'] = pd.to_numeric(df['comptage'], errors='coerce')
    df = df.dropna(subset=['comptage_num'])
    df = df[df['comptage_num'] >= 0]
    print(f"  {len(df):,} lignes valides")
    
    print("\nParsing des coordonnées...")
    coords_split = df['coordonnees'].astype(str).str.split(',', expand=True)
    df['latitude'] = pd.to_numeric(coords_split[0], errors='coerce')
    df['longitude'] = pd.to_numeric(coords_split[1], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"   {len(df):,} lignes avec coordonnées")
    
    print("\nCalcul des arrondissements")
    centroides = np.array([
        [48.8608, 2.3418, 1], [48.8681, 2.3424, 2], [48.8634, 2.3596, 3],
        [48.8567, 2.3612, 4], [48.8458, 2.3497, 5], [48.8503, 2.3318, 6],
        [48.8566, 2.3165, 7], [48.8722, 2.3121, 8], [48.8754, 2.3417, 9],
        [48.8760, 2.3618, 10], [48.8575, 2.3820, 11], [48.8397, 2.3882, 12],
        [48.8322, 2.3665, 13], [48.8338, 2.3268, 14], [48.8407, 2.2862, 15],
        [48.8513, 2.2646, 16], [48.8873, 2.3089, 17], [48.8903, 2.3448, 18],
        [48.8828, 2.3839, 19], [48.8649, 2.3969, 20],
    ])
    
    lats = df['latitude'].values[:, np.newaxis]
    lons = df['longitude'].values[:, np.newaxis]
    distances = np.sqrt((lats - centroides[:, 0])**2 + (lons - centroides[:, 1])**2)
    df['arrondissement'] = centroides[np.argmin(distances, axis=1), 2].astype(int)
    
    print(f"\n{'='*80}")
    print(f"DONNÉES FINALES")
    print(f"{'='*80}")
    print(f"Lignes : {len(df):,}")
    print(f"Période : {df['date_heure_clean'].min()} → {df['date_heure_clean'].max()}")
    print(f"Arrondissements : {sorted(df['arrondissement'].unique())}")
    print(f"Total passages : {df['comptage_num'].sum():,.0f}")
    
    return df


def plot_monthly_evolution(df):
    """
    Graphique 1 : Évolution mensuelle du trafic vélo.
    """
    print("\n" + "="*80)
    print("GRAPHIQUE 1 : ÉVOLUTION MENSUELLE")
    print("="*80)
    
    evolution_mensuelle = df.groupby('annee_mois')['comptage_num'].sum().reset_index()
    evolution_mensuelle['annee_mois_str'] = evolution_mensuelle['annee_mois'].astype(str)
    
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(range(len(evolution_mensuelle)), 
            evolution_mensuelle['comptage_num'] / 1_000_000,
            marker='o', linewidth=2.5, markersize=5, color='#2E86AB')
    ax.fill_between(range(len(evolution_mensuelle)), 
                     evolution_mensuelle['comptage_num'] / 1_000_000,
                     alpha=0.3, color='#2E86AB')
    ax.set_title('Évolution du trafic vélo à Paris par mois (2018-2024)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Passages (millions)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mois', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    step = max(1, len(evolution_mensuelle) // 24)
    ax.set_xticks(range(0, len(evolution_mensuelle), step))
    ax.set_xticklabels([evolution_mensuelle.iloc[i]['annee_mois_str'] 
                        for i in range(0, len(evolution_mensuelle), step)], 
                        rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_top_arrondissements(df):
    """
    Graphique 2 : Top 10 arrondissements.
    """
    print("\n" + "="*80)
    print("GRAPHIQUE 2 : TOP 10 ARRONDISSEMENTS")
    print("="*80)
    
    evolution_arrond = df.groupby(['annee_mois', 'arrondissement'])['comptage_num'].sum().reset_index()
    top_10 = df.groupby('arrondissement')['comptage_num'].sum().nlargest(10).index
    
    print("\nClassement des arrondissements (total 2018-2024) :")
    for i, (arrond, total) in enumerate(df.groupby('arrondissement')['comptage_num'].sum().nlargest(10).items(), 1):
        pct = total / df['comptage_num'].sum() * 100
        print(f"  {i:2d}. {arrond:2d}e arr. : {total:>15,.0f} passages ({pct:5.2f}%)")
    
    fig, ax = plt.subplots(figsize=(20, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, arrond in enumerate(sorted(top_10)):
        data = evolution_arrond[evolution_arrond['arrondissement'] == arrond].sort_values('annee_mois')
        data['annee_mois_str'] = data['annee_mois'].astype(str)
        ax.plot(range(len(data)), data['comptage_num'] / 1_000_000,
                marker='o', linewidth=2.5, label=f'{arrond}e', 
                alpha=0.9, markersize=4, color=colors[idx])
    
    ax.set_title('Évolution du trafic vélo - Top 10 arrondissements', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Passages (millions)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mois', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, 
              title='Arrondissement', title_fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    dates_uniques = sorted(evolution_arrond['annee_mois'].unique())
    dates_str = [str(d) for d in dates_uniques]
    
    step = max(1, len(dates_uniques) // 24)
    ax.set_xticks(range(0, len(dates_uniques), step))
    ax.set_xticklabels([dates_str[i] for i in range(0, len(dates_uniques), step)], 
                        rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_constant_counters(df):
    """
    Graphique 3 : Évolution avec compteurs constants.
    """
    print("\n" + "="*80)
    print("GRAPHIQUE 3 : ÉVOLUTION AVEC COMPTEURS CONSTANTS")
    print("="*80)
    
    print("\nIdentification des compteurs constants")
    
    compteurs_annees = df.groupby('id_compteur')['annee'].apply(lambda x: set(x.unique()))
    
    annees_completes = {2018, 2019, 2020, 2021, 2022, 2023, 2024}
    compteurs_constants = [
        compteur for compteur, annees in compteurs_annees.items() 
        if annees_completes.issubset(annees)
    ]
    
    print(f"  Compteurs totaux : {df['id_compteur'].nunique():,}")
    print(f"  Compteurs constants (2018-2024) : {len(compteurs_constants):,}")
    
    if len(compteurs_constants) == 0:
        print("   Aucun compteur présent sur toute la période")
        
        annees_completes = {2019, 2020, 2021, 2022, 2023, 2024}
        compteurs_constants = [
            compteur for compteur, annees in compteurs_annees.items() 
            if annees_completes.issubset(annees)
        ]
        print(f"  Compteurs constants (2019-2024) : {len(compteurs_constants):,}")
        periode_label = "2019-2024"
    else:
        periode_label = "2018-2024"
    
    if len(compteurs_constants) > 0:
        df_constants = df[df['id_compteur'].isin(compteurs_constants)].copy()
        
        print(f"  Lignes avec compteurs constants : {len(df_constants):,} / {len(df):,}")
        pct_donnees = len(df_constants) / len(df) * 100
        print(f"  Pourcentage des données : {pct_donnees:.1f}%")
        
        evolution_constants = df_constants.groupby('annee_mois')['comptage_num'].sum().reset_index()
        evolution_constants['annee_mois_str'] = evolution_constants['annee_mois'].astype(str)
        
        evolution_totale = df.groupby('annee_mois')['comptage_num'].sum().reset_index()
        evolution_totale['annee_mois_str'] = evolution_totale['annee_mois'].astype(str)
        
        fig, ax = plt.subplots(figsize=(20, 8))
        
        ax.plot(range(len(evolution_totale)), 
                evolution_totale['comptage_num'] / 1_000_000,
                marker='o', linewidth=2.5, markersize=5, color='#2E86AB', 
                label='Tous les compteurs', alpha=0.8)
        ax.fill_between(range(len(evolution_totale)), 
                          evolution_totale['comptage_num'] / 1_000_000,
                          alpha=0.2, color='#2E86AB')
        
        ax.plot(range(len(evolution_constants)), 
                evolution_constants['comptage_num'] / 1_000_000,
                marker='s', linewidth=2.5, markersize=5, color='#E63946',
                label=f'Compteurs constants ({len(compteurs_constants)})', alpha=0.8)
        ax.fill_between(range(len(evolution_constants)), 
                          evolution_constants['comptage_num'] / 1_000_000,
                          alpha=0.2, color='#E63946')
        
        ax.set_title(f'Comparaison : Tous les compteurs vs Compteurs constants ({periode_label})', 
                      fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Passages (millions)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Mois', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        step = max(1, len(evolution_constants) // 24)
        ax.set_xticks(range(0, len(evolution_constants), step))
        ax.set_xticklabels([evolution_constants.iloc[i]['annee_mois_str'] 
                             for i in range(0, len(evolution_constants), step)], 
                            rotation=45, ha='right', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        print("\n Statistiques comparatives :")
        
        premiere_annee = evolution_constants['annee_mois'].min().year
        derniere_annee = evolution_constants['annee_mois'].max().year
        
        print(f"\n  Période analysée : {premiere_annee} - {derniere_annee}")
        
        moyennes_annee = df_constants.groupby('annee')['comptage_num'].sum()
        print(f"\n  Évolution annuelle (compteurs constants) :")
        for annee in sorted(moyennes_annee.index):
            passages = moyennes_annee[annee]
            if annee == moyennes_annee.index.min():
                print(f"    {int(annee)} : {passages:>15,.0f} passages (référence = base 100)")
            else:
                annee_ref = moyennes_annee.index.min()
                indice = (passages / moyennes_annee[annee_ref]) * 100
                croissance = indice - 100
                print(f"    {int(annee)} : {passages:>15,.0f} passages (indice {indice:6.1f}, {croissance:+6.1f}%)")
        
        valeur_debut = moyennes_annee.iloc[0]
        valeur_fin = moyennes_annee.iloc[-1]
        indice_final = (valeur_fin / valeur_debut) * 100
        croissance_totale = indice_final - 100
        nb_annees = len(moyennes_annee) - 1
        croissance_annuelle = ((valeur_fin / valeur_debut) ** (1/nb_annees) - 1) * 100
        
        print(f"\n  Indice final : {indice_final:.1f} (base 100 en {int(moyennes_annee.index.min())})")
        print(f"  Croissance totale : {croissance_totale:+.1f}%")
        print(f"  Croissance annuelle moyenne : {croissance_annuelle:+.1f}%")
        
    else:
        print("Impossible de trouver des compteurs constants sur la période")
    
    print("\n" + "="*80)
    print("ANALYSE COMPTEURS CONSTANTS TERMINÉE")
    print("="*80)








def analyze_commute_traffic(df):
    """
    Analyse la part du vélo-taff (trajets domicile-travail) dans le trafic cyclable.
    """
    print("\n" + "="*80)
    print("ANALYSE VÉLO-TAFF (trajets domicile-travail)")
    print("="*80)
    
    df_velo_taff = df.copy()
    
    df_velo_taff['heure'] = df_velo_taff['date_heure_clean'].dt.hour
    df_velo_taff['mois'] = df_velo_taff['date_heure_clean'].dt.month
    df_velo_taff['jour_semaine'] = df_velo_taff['date_heure_clean'].dt.dayofweek
    
    print("\n Exclusion de juillet et août...")
    nb_avant = len(df_velo_taff)
    df_velo_taff = df_velo_taff[~df_velo_taff['mois'].isin([7, 8])].copy()
    nb_apres = len(df_velo_taff)
    print(f"  Lignes supprimées : {nb_avant - nb_apres:,} ({(nb_avant-nb_apres)/nb_avant*100:.1f}%)")
    print(f"  Lignes restantes : {nb_apres:,}")
    
    df_velo_taff['créneau'] = 'autres'
    df_velo_taff.loc[df_velo_taff['heure'].isin([7, 8, 9]), 'créneau'] = 'matin_taff'
    df_velo_taff.loc[df_velo_taff['heure'].isin([17, 18, 19]), 'créneau'] = 'soir_taff'
    
    df_semaine = df_velo_taff[df_velo_taff['jour_semaine'] < 5].copy()
    
    print("\nCréneaux horaires définis :")
    print("  • Vélo-taff matin : 7h-10h (lun-ven, hors juil-août)")
    print("  • Vélo-taff soir  : 17h-20h (lun-ven, hors juil-août)")
    print("  • Autres          : reste du temps")
    
    evolution_creneaux = df_semaine.groupby(['annee', 'créneau'])['comptage_num'].sum().reset_index()
    pivot_creneaux = evolution_creneaux.pivot(index='annee', columns='créneau', values='comptage_num').fillna(0)
    
    pivot_creneaux['total_taff'] = pivot_creneaux['matin_taff'] + pivot_creneaux['soir_taff']
    pivot_creneaux['total'] = pivot_creneaux['matin_taff'] + pivot_creneaux['soir_taff'] + pivot_creneaux['autres']
    
    pivot_creneaux['pct_taff'] = (pivot_creneaux['total_taff'] / pivot_creneaux['total']) * 100
    pivot_creneaux['pct_autres'] = (pivot_creneaux['autres'] / pivot_creneaux['total']) * 100
    
    pivot_creneaux = pivot_creneaux.reset_index()
    
    total_global = pivot_creneaux[['matin_taff', 'soir_taff', 'autres']].sum()
    print(f"\n Statistiques globales (2018-2024, lun-ven, hors juil-août) :")
    print(f"  • Vélo-taff matin : {total_global['matin_taff']:>15,.0f} passages ({total_global['matin_taff']/total_global.sum()*100:5.1f}%)")
    print(f"  • Vélo-taff soir  : {total_global['soir_taff']:>15,.0f} passages ({total_global['soir_taff']/total_global.sum()*100:5.1f}%)")
    print(f"  • TOTAL vélo-taff : {total_global['matin_taff']+total_global['soir_taff']:>15,.0f} passages ({(total_global['matin_taff']+total_global['soir_taff'])/total_global.sum()*100:5.1f}%)")
    print(f"  • Autres créneaux : {total_global['autres']:>15,.0f} passages ({total_global['autres']/total_global.sum()*100:5.1f}%)")
    
    return pivot_creneaux


def plot_commute_evolution(pivot_creneaux):
    """
    Graphique : Évolution annuelle de la part du vélo-taff.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    annees = pivot_creneaux['annee'].values
    
    ax.plot(annees, pivot_creneaux['pct_taff'],
            marker='o', linewidth=3, markersize=10, color='#E63946', 
            label='Vélo-taff (7-10h + 17-20h)', zorder=3)
    ax.plot(annees, pivot_creneaux['pct_autres'],
            marker='s', linewidth=3, markersize=10, color='#457B9D', 
            label='Autres créneaux', zorder=3)
    
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='50%')
    ax.fill_between(annees, pivot_creneaux['pct_taff'], alpha=0.3, color='#E63946')
    ax.fill_between(annees, pivot_creneaux['pct_autres'], alpha=0.3, color='#457B9D')
    
    ax.set_title('Évolution annuelle de la part du vélo-taff\n(lun-ven, hors juillet-août)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Pourcentage du trafic (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Année', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    ax.set_xticks(annees)
    ax.set_xticklabels([int(a) for a in annees], fontsize=12)
    
    for i, annee in enumerate(annees):
        ax.annotate(f"{pivot_creneaux.iloc[i]['pct_taff']:.1f}%", 
                    xy=(annee, pivot_creneaux.iloc[i]['pct_taff']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold',
                    color='#E63946')
    
    plt.tight_layout()
    plt.show()
    
    print("\n Évolution détaillée par année :")
    print(f"\n{'Année':<8} {'Vélo-taff':>15} {'Autres':>15} {'Total':>15} {'% Taff':>10}")
    print("-" * 70)
    for _, row in pivot_creneaux.iterrows():
        print(f"{int(row['annee']):<8} {row['total_taff']:>15,.0f} {row['autres']:>15,.0f} "
              f"{row['total']:>15,.0f} {row['pct_taff']:>9.1f}%")
    
    print("\nAnalyse vélo-taff terminée")




def analyze_telework_effect(df):
    """
    Analyse l'effet du télétravail : Lundi+Vendredi vs Mardi+Mercredi+Jeudi.
    """
    print("\n" + "="*80)
    print("ANALYSE TÉLÉTRAVAIL : Lundi+Vendredi vs Mardi+Mercredi+Jeudi")
    print("="*80)
    
    df_teletravail = df.copy()
    
    df_teletravail['heure'] = df_teletravail['date_heure_clean'].dt.hour
    df_teletravail['mois'] = df_teletravail['date_heure_clean'].dt.month
    df_teletravail['jour_semaine'] = df_teletravail['date_heure_clean'].dt.dayofweek
    
    print("\nFiltrage des données")
    df_teletravail = df_teletravail[
        (~df_teletravail['mois'].isin([7, 8])) &
        (df_teletravail['jour_semaine'] < 5)
    ].copy()
    
    df_teletravail = df_teletravail[
        df_teletravail['heure'].isin([7, 8, 9, 17, 18, 19])
    ].copy()
    
    print(f"  Lignes après filtrage : {len(df_teletravail):,}")
    
    df_teletravail['type_jour'] = 'milieu_semaine'
    df_teletravail.loc[df_teletravail['jour_semaine'].isin([0, 4]), 'type_jour'] = 'debut_fin_semaine'
    
    print("\nCatégorisation des jours :")
    print("  • Début/Fin semaine : Lundi + Vendredi")
    print("  • Milieu semaine    : Mardi + Mercredi + Jeudi")
    print("  • Heures analysées  : 7-10h + 17-20h")
    print("  • Période           : Hors juillet-août")
    
    evolution_jours = df_teletravail.groupby(['annee', 'type_jour'])['comptage_num'].sum().reset_index()
    pivot_jours = evolution_jours.pivot(index='annee', columns='type_jour', values='comptage_num').fillna(0)
    
    pivot_jours['ratio'] = pivot_jours['debut_fin_semaine'] / pivot_jours['milieu_semaine']
    
    pivot_jours['trafic_moyen_debut_fin'] = pivot_jours['debut_fin_semaine'] / 2
    pivot_jours['trafic_moyen_milieu'] = pivot_jours['milieu_semaine'] / 3
    pivot_jours['ratio_par_jour'] = pivot_jours['trafic_moyen_debut_fin'] / pivot_jours['trafic_moyen_milieu']
    
    pivot_jours = pivot_jours.reset_index()
    
    print("\nRésultats par année :")
    print(f"\n{'Année':<8} {'Lun+Ven':>15} {'Mar+Mer+Jeu':>15} {'Ratio brut':>12} {'Ratio/jour':>12}")
    print("-" * 70)
    
    for _, row in pivot_jours.iterrows():
        annee = int(row['annee'])
        lun_ven = row['debut_fin_semaine']
        milieu = row['milieu_semaine']
        ratio_brut = row['ratio']
        ratio_jour = row['ratio_par_jour']
        
        print(f"{annee:<8} {lun_ven:>15,.0f} {milieu:>15,.0f} {ratio_brut:>12.3f} {ratio_jour:>12.3f}")
    
    premiere_annee = pivot_jours.iloc[0]
    derniere_annee = pivot_jours.iloc[-1]
    evolution_ratio = ((derniere_annee['ratio_par_jour'] / premiere_annee['ratio_par_jour']) - 1) * 100
    
    print(f"\nÉvolution du ratio par jour :")
    print(f"  {int(premiere_annee['annee'])} : {premiere_annee['ratio_par_jour']:.3f}")
    print(f"  {int(derniere_annee['annee'])} : {derniere_annee['ratio_par_jour']:.3f}")
    print(f"  Variation : {evolution_ratio:+.1f}%")
    
    if evolution_ratio < -5:
        print(f"\n Baisse significative du ratio → Effet télétravail probable")
    elif evolution_ratio < 0:
        print(f"\n  → Légère baisse du ratio → Effet télétravail possible")
    else:
        print(f"\n  → Pas d'effet télétravail détecté (ratio stable ou en hausse)")
    
    return pivot_jours


def plot_telework_analysis(pivot_jours):
    """
    Graphique : Évolution du ratio Lun+Ven / Mar+Mer+Jeu.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    annees = pivot_jours['annee'].values
    
    width = 0.35
    x = np.arange(len(annees))
    
    bars1 = ax1.bar(x - width/2, pivot_jours['trafic_moyen_debut_fin'] / 1_000_000, 
                    width, label='Lun+Ven (moyenne/jour)', color='#E63946', alpha=0.8)
    bars2 = ax1.bar(x + width/2, pivot_jours['trafic_moyen_milieu'] / 1_000_000, 
                    width, label='Mar+Mer+Jeu (moyenne/jour)', color='#457B9D', alpha=0.8)
    
    ax1.set_title('Comparaison du trafic moyen par jour : Lun+Ven vs Mar+Mer+Jeu\n(Heures de bureau 7-10h + 17-20h, hors juillet-août)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Passages moyens par jour (millions)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Année', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([int(a) for a in annees])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1,
                 f'{height1:.2f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2,
                 f'{height2:.2f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.plot(annees, pivot_jours['ratio_par_jour'],
             marker='o', linewidth=3, markersize=12, color='#2A9D8F', 
             label='Ratio (Lun+Ven) / (Mar+Mer+Jeu)', zorder=3)
    ax2.fill_between(annees, pivot_jours['ratio_par_jour'], alpha=0.3, color='#2A9D8F')
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
                label='Ratio = 1 (pas de différence)')
    
    ax2.set_title('Ratio de trafic : (Lun+Ven) / (Mar+Mer+Jeu)\n(Ratio < 1 = moins de trafic en début/fin de semaine)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Ratio', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Année', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(annees)
    ax2.set_xticklabels([int(a) for a in annees])
    
    for i, annee in enumerate(annees):
        ratio = pivot_jours.iloc[i]['ratio_par_jour']
        ax2.annotate(f"{ratio:.3f}", 
                    xy=(annee, ratio),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold',
                    color='#2A9D8F')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("ANALYSE TÉLÉTRAVAIL TERMINÉE")
    print("="*80)
    print("\n Interprétation :")
    print("  • Ratio > 1 : Plus de trafic le lun+ven que mar+mer+jeu")
    print("  • Ratio < 1 : Moins de trafic le lun+ven (effet télétravail ?)")
    print("  • Ratio qui baisse = augmentation du télétravail")