import os
import pandas as pd
import random
import numpy as np
import time
from typing import Dict, List, Tuple

from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font

from solve import solve_multi

MAX_DIST = 10

ascii_art = r"""

                                          
  ____    _    ____  ____ ___ _____ _   _ 
 / ___|  / \  |  _ \|  _ |_ _| ____| \ | |
| |  _  / _ \ | |_) | | | | ||  _| |  \| |
| |_| |/ ___ \|  _ <| |_| | || |___| |\  |
 \____/_/   \_|_| \_|____|___|_____|_| \_|
                                          

"""

couleur_gardien = '\033[35m'

def print_ascii():
    print(f"\033[1m\033[31m{ascii_art}\033[0m")

def main():
    dir = input(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m Répertoire où se trouvent les fichiers Excel (.xlsx) : ")
    dir = dir.strip()
    dir = dir.replace("\\ ", " ")
    dir = os.path.abspath(dir)

    verbose = True

    print()

    try:
        excel_files = [
            f for f in os.listdir(dir) 
            if f.endswith('.xlsx')  # Ne garder que les fichiers Excel
            and not f.startswith('~')  # Exclure les fichiers temporaires Excel (~$...)
            and not f.startswith('.')  # Exclure les fichiers cachés (.fichier)
            and not f.startswith('$')  # Exclure d'autres fichiers temporaires potentiels
        ]
        random.shuffle(excel_files)
        assert len(excel_files) > 0, f"Aucun fichier .xlsx trouvés dans {dir}. Chemin d'exécution: {os.getcwd()}"
    except FileNotFoundError:
        print(f"\033[1m\033[31m[ERREUR]\033[0m Répertoire non trouvé : \033[1m\033[33m{dir}\033[0m")
        return
    except PermissionError:
        print(f"\033[1m\033[31m[ERREUR]\033[0m Accès refusé au répertoire : \033[1m\033[33m{dir}\033[0m")
        return
    except Exception as e:
        print(f"\033[1m\033[31m[ERREUR]\033[0m Une erreur inattendue s'est produite : \033[1m\033[33m{str(e)}\033[0m")
        return

    if len(excel_files) == 1:
        print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m \033[1m{len(excel_files)}\033[0m fichier trouvé: \033[1m\033[33m{', '.join(excel_files)}\033[0m")
    else:
        print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m \033[1m{len(excel_files)}\033[0m fichiers trouvés: \033[1m\033[33m{', '.join(excel_files)}\033[0m")

    eqs_to_global = []
    global_to_eqs = []
    preferences_eqs = []
    reductions_eqs = []
    attributs_eqs: List[Dict[str, List[bool]]] = [] # Pour stocker les attributs de chaque équipe
    Ns = []
    Ds = []
    mdc_eqs = []

    global_mdc = set()

    planning_initiaux = []  # Liste pour stocker les plannings initiaux
    jours_a_modifier = []  # Liste pour stocker les indices des jours à modifier
    doit_modifier = []  # Liste pour savoir si on doit modifier ou créer pour chaque équipe
    skip_optims = []

    # Lire chaque fichier Excel et collecter les données
    for file in excel_files:
        df = pd.read_excel(os.path.join(dir, file))
        df = df.drop(df.columns[:3], axis=1)
        df = df.fillna(0)

        mdc = df.columns.to_list()
        preferences = df.to_numpy().T
        N, D = preferences.shape

        wb = load_workbook(os.path.join(dir, file))
        ws = wb.worksheets[0]

        # Lire le planning initial (colonnes garde et astreinte)
        gardes = []
        astreintes = []
        jours_gras = {'garde': [], 'astreinte': [], 'jour_debut_modif': -1}  # Pour stocker les indices des cellules en gras
        planning_vide = True

        # Ajouter cette partie pour lire le premier jour modifiable
        for row in range(2, ws.max_row + 1):  # Skip header row
            if ws.cell(row=row, column=1).font.bold:  # Colonne A
                jours_gras['jour_debut_modif'] = row - 2  # -2 car on commence à la ligne 2
                break
        
        for row in range(2, ws.max_row + 1):  # Skip header row
            garde = ws.cell(row=row, column=2).value  # Colonne B
            astreinte = ws.cell(row=row, column=3).value  # Colonne C
            
            # Vérifier si les cellules sont en gras
            if ws.cell(row=row, column=2).font.bold:
                jours_gras['garde'].append(row-2)  # -2 car on commence à la ligne 2
            if ws.cell(row=row, column=3).font.bold:
                jours_gras['astreinte'].append(row-2)
            
            # Si une des cases n'est pas vide, on note que le planning n'est pas vide
            if garde is not None or astreinte is not None:
                planning_vide = False
            
            # Convertir les noms en indices (on utilise -1 pour les cases vides)
            garde_idx = next((i for i, name in enumerate(mdc) if name == garde), -1) if garde else -1
            astreinte_idx = next((i for i, name in enumerate(mdc) if name == astreinte), -1) if astreinte else -1
            gardes.append(garde_idx)
            astreintes.append(astreinte_idx)
        
        planning_initial = gardes + astreintes if not planning_vide else None
        
        planning_initiaux.append(planning_initial)
        jours_a_modifier.append(jours_gras)
        doit_modifier.append(not planning_vide)

        # Vérifier si toutes les cases sont remplies et s'il n'y a pas de cellules en gras
        toutes_cases_remplies = all(garde is not None and astreinte is not None 
                                  for garde, astreinte in zip(ws.iter_rows(min_row=2, max_col=2, min_col=2),
                                                            ws.iter_rows(min_row=2, max_col=3, min_col=3)))
        aucune_case_gras = not (any(ws.cell(row=row, column=2).font.bold or ws.cell(row=row, column=3).font.bold 
                                   for row in range(2, ws.max_row + 1)))
        
        wb.close()
        
        skip_optim = not planning_vide and toutes_cases_remplies and aucune_case_gras
        skip_optims.append(skip_optim)
        
        if skip_optim and verbose:
            print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m \033[1m\033[34m{file}\033[0m : Planning complet et sans modification demandée -> pas d'optimisation")

        # nouvelle partie : tentative de lecture de la feuille "attributs"
        try:
            df_attributs = pd.read_excel(os.path.join(dir, file), sheet_name='attributs')

            mdc_attributs = df_attributs.columns.tolist()[1:]
            if mdc != mdc_attributs:
                # Trouver les différences pour un message d'erreur plus informatif
                mdc_manquants = set(mdc) - set(mdc_attributs)
                mdc_supplementaires = set(mdc_attributs) - set(mdc)
                
                error_msg = f"\033[1m\033[31m[ERREUR]\033[0m Dans le fichier \033[1m\033[33m{file}\033[0m :"
                if mdc_manquants:
                    error_msg += f"\nMédecins présents dans la feuille principale mais absents de la feuille attributs : \033[1m\033[36m{', '.join(mdc_manquants)}\033[0m"
                if mdc_supplementaires:
                    error_msg += f"\nMédecins présents dans la feuille attributs mais absents de la feuille principale : \033[1m\033[36m{', '.join(mdc_supplementaires)}\033[0m"
                
                print(error_msg)
                return  # Arrêt du programme

            attributs_eq = {}

            for _, row in df_attributs.iterrows():
                nom_attribut = row.iloc[0]
                valeurs = row.iloc[1:].astype(bool).tolist()
                attributs_eq[nom_attribut] = valeurs
                
            attributs_eqs.append(attributs_eq)

            if verbose:
                print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m Attributs trouvés pour \033[1m\033[34m{file}\033[0m :")
                for attr, vals in attributs_eq.items():
                    medecins_avec_attr = [m for m, v in zip(mdc, vals) if v]
                    print(f"  - {attr}: {', '.join(medecins_avec_attr)}")
        except ValueError:
            attributs_eqs.append({})

        Ns.append(N)
        Ds.append(D)
        preferences_eqs.append(preferences)
        mdc_eqs.append(mdc)

        global_mdc.update(mdc)

    global_mdc = list(global_mdc)

    for i, (planning_initial, jours_gras, doit_modif) in enumerate(zip(planning_initiaux, jours_a_modifier, doit_modifier)):
        if not doit_modif:  # Si c'était un planning vide, on skip
            continue
        
        # On vérifie s'il y a des modifications à faire
        if not jours_gras['garde'] and not jours_gras['astreinte']:
            continue
        
        debut_modif_str = f" (à partir du jour \033[1m{jours_gras['jour_debut_modif'] + 1}\033[0m)" if jours_gras['jour_debut_modif'] != -1 else ""
        print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m Modifications demandées pour \033[1m\033[34mÉquipe {i+1}\033[0m{debut_modif_str} :")

        if jours_gras['garde']:
            print("  - Gardes à modifier aux jours :", end=" ")
            print(", ".join([f"\033[1m{j+1}\033[0m" for j in jours_gras['garde']]))
            
        if jours_gras['astreinte']:
            print("  - Astreintes à modifier aux jours :", end=" ")
            print(", ".join([f"\033[1m{j+1}\033[0m" for j in jours_gras['astreinte']]))
        
        if jours_gras['jour_debut_modif'] != -1:
            modifications_precoces = []
            
            # Vérifier les gardes
            for jour in jours_gras['garde']:
                if jour < jours_gras['jour_debut_modif']:
                    modifications_precoces.append(f"Garde au jour {jour + 1}")
            
            # Vérifier les astreintes
            for jour in jours_gras['astreinte']:
                if jour < jours_gras['jour_debut_modif']:
                    modifications_precoces.append(f"Astreinte au jour {jour + 1}")
            
            if modifications_precoces:
                print(f"\033[1m\033[31m[ATTENTION]\033[0m Des modifications sont demandées avant le jour \033[1m{jours_gras['jour_debut_modif'] + 1}\033[0m :")
                for modif in modifications_precoces:
                    print(f"  - {modif}")

    if any(bool(jours_gras['garde']) or bool(jours_gras['astreinte']) for jours_gras in jours_a_modifier if jours_gras):
        print()

    print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m \033[1m{len(global_mdc)}\033[0m médecins uniques trouvés: ", end="")
    print(", ".join([f"\033[1m\033[36m{mdc}\033[0m" for mdc in global_mdc]))

    for i, (mdc_eq, file) in enumerate(zip(mdc_eqs, excel_files), 1):
        print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m \033[1m\033[34mÉquipe {i}\033[0m (\033[1m\033[33m{file}\033[0m) : ", end="")
        print(", ".join([f"\033[1m\033[36m{mdc}\033[0m" for mdc in mdc_eq]))

    print()
    print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m Début de l'optimisation.")

    # Mapper chaque médecin à une position globale
    for mdc_eq in mdc_eqs:
        eq_to_global = {i: global_mdc.index(member) for i, member in enumerate(mdc_eq)}
        global_to_eq = {global_mdc.index(member): i for i, member in enumerate(mdc_eq)}

        eqs_to_global.append(eq_to_global)
        global_to_eqs.append(global_to_eq)

    # Calculer les réductions pour chaque équipe
    nombre_equipes_global = {idx: sum([1 for mdc_eq in mdc_eqs if member in mdc_eq]) for idx, member in enumerate(global_mdc)}

    for mdc_eq, global_to_eq in zip(mdc_eqs, global_to_eqs):
        reductions = [None] * len(mdc_eq)
        for mdc in nombre_equipes_global:
            if mdc in global_to_eq:
                reductions[global_to_eq[mdc]] = nombre_equipes_global[mdc]
        reductions_eqs.append(np.array(reductions))

    # Appel à la fonction de résolution multi-équipes
    resultat_eqs, score_final_eqs = solve_multi(Ns, Ds, preferences_eqs, reductions_eqs, attributs_eqs, eqs_to_global, global_to_eqs, planning_initiaux, jours_a_modifier, skip_optims)

    print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m Scores finaux par équipe : ", end="")
    print(", ".join([f"\033[1m\033[34mÉquipe {i+1}\033[0m : \033[1m\033[36m{score:.0f}\033[0m" for i, score in enumerate(score_final_eqs)]))

    time.sleep(1)

    print()
    for i, (resultat_eq, planning_initial, doit_modif) in enumerate(zip(resultat_eqs, planning_initiaux, doit_modifier)):
        if not doit_modif:  # Si c'était une création de planning, on skip
            continue
            
        # Calculer la distance (en ne prenant en compte que les cases non vides du planning initial)
        masque = np.array(planning_initial) != -1
        distance = np.sum(np.array(resultat_eq)[masque] != np.array(planning_initial)[masque])
        
        print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m \033[1m\033[34mÉquipe {i+1}\033[0m : \033[1m{distance}\033[0m changements par rapport au planning initial")
        
        if verbose:
            D = Ds[i]
            for jour in range(D):
                # Vérifier les changements de garde
                if planning_initial[jour] != -1 and resultat_eq[jour] != planning_initial[jour]:
                    ancien_mdc = mdc_eqs[i][planning_initial[jour]]
                    nouveau_mdc = mdc_eqs[i][resultat_eq[jour]]
                    print(f"  - Jour {jour+1}, garde : \033[1m\033[36m{ancien_mdc}\033[0m → \033[1m\033[36m{nouveau_mdc}\033[0m")
                
                # Vérifier les changements d'astreinte
                if planning_initial[D + jour] != -1 and resultat_eq[D + jour] != planning_initial[D + jour]:
                    ancien_mdc = mdc_eqs[i][planning_initial[D + jour]]
                    nouveau_mdc = mdc_eqs[i][resultat_eq[D + jour]]
                    print(f"  - Jour {jour+1}, astreinte : \033[1m\033[36m{ancien_mdc}\033[0m → \033[1m\033[36m{nouveau_mdc}\033[0m")

    time.sleep(1)

    check_coherence(resultat_eqs, Ds, global_mdc, eqs_to_global)

    time.sleep(1)

    total_jours_problemes = 0
    for i, (resultat_eq, preferences_eq) in enumerate(zip(resultat_eqs, preferences_eqs)):
        jours_problemes_eq = 0
        for jour in range(Ds[i]):
            mdc_garde = resultat_eq[jour]
            preference = preferences_eq[mdc_garde, jour]

            if preference < 0:
                jours_problemes_eq += 1
                total_jours_problemes += 1
                if verbose:
                    if total_jours_problemes == 1:
                        print(f"\033[1m\033[31m[GARDIEN]\033[0m Jours problématiques (médecins affectés avec préférence négative) :")
                        print(f"\033[1m\033[34mÉquipe {i+1}\033[0m :")
                    print(f"  - Jour {jour+1}: Médecin \033[1m\033[36m{mdc_eqs[i][mdc_garde]}\033[0m affecté avec préférence de \033[1m\033[31m{preference}\033[0m")

    print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m Vérification des jours problématiques...")
    if total_jours_problemes > 0:
        print(f"\033[1m\033[31m[GARDIEN]\033[0m Total de jours problématiques (préférence non respectée): \033[1m\033[31m{total_jours_problemes}\033[0m")
    else:
        print(f"\033[1m\033[32m[GARDIEN]\033[0m Total de jours problématiques (préférence non respectée): \033[1m\033[32m{total_jours_problemes}\033[0m")

    time.sleep(1)
    print()
    print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m Vérification des attributs des médecins...")

    total_jours_sans_attributs = 0
    for i, (resultat_eq, attributs_eq) in enumerate(zip(resultat_eqs, attributs_eqs)):
        if not attributs_eq:  # Si pas d'attributs pour cette équipe, on passe
            continue

        jours_problemes_eq = 0
        for jour in range(Ds[i]):
            mdc_garde = resultat_eq[jour]
            mdc_astreinte = resultat_eq[Ds[i] + jour]
            
            for nom_attribut, mdc_avec_attribut in attributs_eq.items():
                if not (mdc_avec_attribut[mdc_garde] or mdc_avec_attribut[mdc_astreinte]):
                    jours_problemes_eq += 1
                    total_jours_sans_attributs += 1
                    if verbose:
                        if total_jours_sans_attributs == 1:
                            print(f"\033[1m\033[34mÉquipe {i+1}\033[0m :")
                        print(f"  - Jour {jour+1}: Attribut \033[1m\033[35m{nom_attribut}\033[0m manquant")
                        print(f"    Garde: \033[1m\033[36m{mdc_eqs[i][mdc_garde]}\033[0m")
                        print(f"    Astreinte: \033[1m\033[36m{mdc_eqs[i][mdc_astreinte]}\033[0m")

    if total_jours_sans_attributs > 0:
        print(f"\033[1m\033[31m[GARDIEN]\033[0m Total de jours avec attributs manquants: \033[1m\033[31m{total_jours_sans_attributs}\033[0m")
    else:
        print(f"\033[1m\033[32m[GARDIEN]\033[0m Tous les jours ont les attributs requis.")

    time.sleep(1)
    print()
    print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m Enregistrement des plannings en cours...")

    # Exporter les résultats dans chaque fichier Excel avec le même nom d'origine
    blue_fill = PatternFill(start_color="6c9beb", end_color="6c9beb", fill_type="solid")
    red_fill = PatternFill(start_color="DB7C6C", end_color="DB7C6C", fill_type="solid")
    green_fill = PatternFill(start_color="99c57a", end_color="99c57a", fill_type="solid")
    gray_fill = PatternFill(start_color="9a9796", end_color="9a9796", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFEB99", end_color="FFEB99", fill_type="solid")
    collision_fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
    jour_off_fill = PatternFill(start_color="FFB366", end_color="FFB366", fill_type="solid")
    attribut_fill = PatternFill(start_color="FFA07A", end_color="FFA07A", fill_type="solid")
    modification_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")

    # Créer une matrice de planning global pour détecter les collisions
    schedule = np.zeros((len(global_mdc), max(Ds)), dtype=int)
    collisions = {i: set() for i in range(len(excel_files))}  # jours avec collision par équipe
    jours_off = {i: set() for i in range(len(excel_files))}   # jours avec problème de jour off
    jours_attributs = {i: set() for i in range(len(excel_files))}  # jours avec attribut manquant

    # Remplir la matrice et détecter les collisions
    for i, (resultat_eq, eq_to_global) in enumerate(zip(resultat_eqs, eqs_to_global)):
        planning_gardes = resultat_eq[:Ds[i]]
        planning_astreintes = resultat_eq[Ds[i]:]

        # Vérifier les collisions
        for day, mdc_local in enumerate(planning_gardes):
            mdc_global = eq_to_global[mdc_local]
            if schedule[mdc_global, day] != 0:
                collisions[i].add(day)
            schedule[mdc_global, day] = 1

        for day, mdc_local in enumerate(planning_astreintes):
            mdc_global = eq_to_global[mdc_local]
            if schedule[mdc_global, day] != 0:
                collisions[i].add(day)
            schedule[mdc_global, day] = 2

        # Vérifier les jours off
        for day in range(Ds[i] - 1):
            if planning_gardes[day] == planning_gardes[day + 1] or \
            planning_gardes[day] == planning_astreintes[day + 1]:
                jours_off[i].add(day + 1)

        # Vérifier les attributs
        if attributs_eqs[i]:
            for jour in range(Ds[i]):
                mdc_garde = planning_gardes[jour]
                mdc_astreinte = planning_astreintes[jour]
                for nom_attribut, mdc_avec_attribut in attributs_eqs[i].items():
                    if not (mdc_avec_attribut[mdc_garde] or mdc_avec_attribut[mdc_astreinte]):
                        jours_attributs[i].add(jour)

    fichiers_crees = []
    for i, file in enumerate(excel_files):
        wb = load_workbook(os.path.join(dir, file))

        # création de la feuille où l'on place les légendes
        if "Légende" not in wb.sheetnames:
            ws_legende = wb.create_sheet("Légende")
        else:
            ws_legende = wb["Légende"]

        # Style pour les titres
        title_font = Font(bold=True, size=12)
        subtitle_font = Font(bold=True)

        # Ajouter les titres
        ws_legende['A1'] = "LÉGENDE DES COULEURS"
        ws_legende['A1'].font = title_font
        
        # Section 1: Problèmes graves
        ws_legende['A3'] = "Problèmes graves :"
        ws_legende['A3'].font = subtitle_font
        
        ws_legende['B4'] = "Collision (médecin assigné à plusieurs équipes)"
        ws_legende['A4'].fill = collision_fill
        
        ws_legende['B5'] = "Jour OFF après garde non respecté"
        ws_legende['A5'].fill = jour_off_fill
        
        ws_legende['B6'] = "Attribut manquant ce jour-là"
        ws_legende['A6'].fill = attribut_fill

        # Section 2: Préférences
        ws_legende['A8'] = "Préférences :"
        ws_legende['A8'].font = subtitle_font
        
        ws_legende['B9'] = "Préférence positive (assignée à une garde)"
        ws_legende['A9'].fill = blue_fill
        
        ws_legende['B10'] = "Préférence négative (assignée à une garde)"
        ws_legende['A10'].fill = yellow_fill
        
        ws_legende['B11'] = "Préférence positive"
        ws_legende['A11'].fill = green_fill
        
        ws_legende['B12'] = "Préférence négative"
        ws_legende['A12'].fill = red_fill
        
        ws_legende['B13'] = "Préférence nulle"
        ws_legende['A13'].fill = gray_fill

        # Section 3: Modifications
        ws_legende['A15'] = "Modifications :"
        ws_legende['A15'].font = subtitle_font
        
        ws_legende['B16'] = "Garde ou astreinte modifiée par rapport au planning initial"
        ws_legende['A16'].fill = modification_fill

        ws_legende.column_dimensions['B'].width = 50
        ws_legende.column_dimensions['A'].width = 4

        for k in range(1, 17):
            ws_legende.row_dimensions[k].height = 20

        # modification de la feuille principale
        ws = wb.worksheets[0]

        for day in range(Ds[i]):
            mdc_garde = resultat_eqs[i][day]
            mdc_astreinte = resultat_eqs[i][Ds[i]:][day]

            # Colorer les gardes/astreintes selon les problèmes détectés
            garde_cell = ws.cell(row=day+2, column=2)
            astreinte_cell = ws.cell(row=day+2, column=3)
            jour_cell = ws.cell(row=day+2, column=1)

            if day in collisions[i]:
                jour_cell.fill = attribut_fill
            elif day in jours_off[i]:
                jour_cell.fill = attribut_fill
            if day in jours_attributs[i]:
                jour_cell.fill = attribut_fill

            # Colorer les changements si c'est une modification de planning
            if planning_initiaux[i] is not None:
                if planning_initiaux[i][day] != -1 and mdc_garde != planning_initiaux[i][day]:
                    garde_cell.fill = modification_fill
                if planning_initiaux[i][Ds[i] + day] != -1 and mdc_astreinte != planning_initiaux[i][Ds[i] + day]:
                    astreinte_cell.fill = modification_fill
            
            # remplir et colorer les préférences
            for j in range(Ns[i]):
                pref_cell = ws.cell(row=day+2, column=4+j)
                preference = pref_cell.value
                
                if preference is None:
                    pref_cell.fill = gray_fill
                elif preference < 0:
                    if j == mdc_garde:  # Si le médecin est assigné à la garde
                        pref_cell.fill = yellow_fill  # Préférence négative assignée
                    else:
                        pref_cell.fill = red_fill
                elif preference > 0:
                    if j == mdc_garde:  # Si le médecin est assigné à la garde
                        pref_cell.fill = blue_fill
                    else:
                        pref_cell.fill = green_fill
                else:
                    pref_cell.fill = gray_fill

            # Mettre à jour les valeurs
            garde_cell.value = mdc_eqs[i][mdc_garde]
            astreinte_cell.value = mdc_eqs[i][mdc_astreinte]

            # Enlever le gras
            garde_cell.font = Font(bold=False)
            astreinte_cell.font = Font(bold=False)
            ws.cell(row=day+2, column=1).font = Font(bold=False)

        output_filename = os.path.join(dir, f"{os.path.splitext(file)[0]}_resultat.xlsx")
        wb.save(output_filename)
        wb.close()
        fichiers_crees.append(f"{os.path.splitext(file)[0]}_resultat.xlsx")

    print(f"\033[1m\033[32m[GARDIEN]\033[0m Enregistrement réussi: \033[1m\033[33m{', '.join([f'{fichier}' for fichier in fichiers_crees])}\033[0m")

def check_coherence(resultat_eqs, Ds, global_mdc, eqs_to_global):
    # Vérification des collisions et du respect des jours OFF après les gardes
    print()
    print(f"\033[1m{couleur_gardien}[GARDIEN]\033[0m Vérification des collisions et des jours OFF après les gardes...")

    # Créer une matrice de planning pour chaque médecin (en IDs globaux) sur l'ensemble des équipes
    schedule = np.zeros((len(global_mdc), max(Ds)), dtype=int)

    detecte = False

    # Remplir la matrice de planning avec les gardes et astreintes
    for i, (resultat_eq, eq_to_global) in enumerate(zip(resultat_eqs, eqs_to_global)):
        planning_gardes = resultat_eq[:Ds[i]]  # première moitié pour les gardes
        planning_astreintes = resultat_eq[Ds[i]:]  # seconde moitié pour les astreintes

        # Remplir avec les gardes
        for day, mdc_local in enumerate(planning_gardes):
            mdc_global = eq_to_global[mdc_local]  # Conversion en ID global

            if schedule[mdc_global, day] != 0:
                print(f"\033[1m\033[31m[ERREUR]\033[0m Collision pour le médecin \033[1m\033[36m{global_mdc[mdc_global]}\033[0m au jour {day+1}")
                detecte = True
            schedule[mdc_global, day] = 1  # Garde

        # Remplir avec les astreintes
        for day, mdc_local in enumerate(planning_astreintes):
            mdc_global = eq_to_global[mdc_local]  # Conversion en ID global

            if schedule[mdc_global, day] != 0:
                print(f"\033[1m\033[31m[ERREUR]\033[0m Collision pour le médecin \033[1m\033[36m{global_mdc[mdc_global]}\033[0m au jour {day+1}")
                detecte = True
            schedule[mdc_global, day] = 2  # Astreinte

    # Vérifier que les médecins ne travaillent pas le jour après une garde
    for mdc_global in range(len(global_mdc)):
        for day in range(max(Ds) - 1):  # On s'arrête un jour avant la fin
            if schedule[mdc_global, day] == 1 and schedule[mdc_global, day + 1] != 0:
                print(f"\033[1m\033[31m[ERREUR]\033[0m Le médecin \033[1m\033[36m{global_mdc[mdc_global]}\033[0m travaille le jour {day+2} après une garde au jour {day+1}")
                detecte = True

    if not detecte:
        print("\033[1m\033[32m[GARDIEN]\033[0m Aucun problème de collision ou de jour OFF détecté.")
    print()

if __name__ == "__main__":
    print_ascii()
    time.sleep(1)

    main()
