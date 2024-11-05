import copy
import numpy as np

from definition import GestionnairePlanning
from algo_ant_colony import recherche_ant_colony
from algo_tabou import recherche_tabou
from config import *

def solve_mono(nombre_jours, nombre_mdc, preferences, reductions=None, attributs=None, eq=None, planning_initial=None, jours_gras=None, skip_optim=False):
    if skip_optim and planning_initial is not None:
        return np.array(planning_initial), 0
    
    gplan = GestionnairePlanning(nombre_mdc, nombre_jours, preferences, reductions, attributs, jours_gras, planning_initial)

    if planning_initial is not None:
        planning_initial = np.array(planning_initial)
        nb_vides = np.sum(np.array(planning_initial) == -1)
        
        # Compléter les cases vides (-1) avec des valeurs aléatoires
        for i in range(len(planning_initial)):
            if planning_initial[i] == -1:
                planning_initial[i] = gplan.random_mdc()
        
        # Forcer les contraintes pour les cases qu'on vient de remplir
        planning_initial = gplan.forcer_contrainte(planning_initial)
        
        # Calculer max_dist en fonction du nombre de changements demandés
        nb_gras = len(jours_gras['garde']) + len(jours_gras['astreinte'])
        max_dist = MAX_DIST + nb_gras + nb_vides
    else:
        planning_initial = None
        max_dist = None

    # PREMIERE ETAPE : ANT COLONY
    resultat_aoc, meilleur_score, scores = recherche_ant_colony(NUM_ANTS, NUM_ITERS_AC, ALPHA, BETA, RHO, gplan, eq=eq, sol_initiale=planning_initial, jours_gras=jours_gras)
    #print(f"Critère final obtenu: {scores[-1]}. Best critère obtenu: {meilleur_score}")
    #print(f"Soft critère: {gplan.calcule_soft_critere(resultat_aoc)}")
    #print(gplan.infos_planning(resultat_aoc))
    #print(f"Respecte les contraintes dures : {not gplan.detecte_contrainte(resultat_aoc)}")

    # DEUXIEME ETAPE : TABOU
    resultat_tabou, scores = recherche_tabou(NUM_ITERS_T, NUM_VOISINS, MAX_STAGNATION, LEN_TABOU, gplan, sol=resultat_aoc, eq=eq, max_dist=max_dist, planning_initial=planning_initial, jours_gras=jours_gras)
    #print(f"Critère final obtenu: {scores[-1]}")
    #print(f"Soft critère: {gplan.calcule_soft_critere(resultat_tabou)}")
    #print(gplan.infos_planning(resultat_tabou))
    #print(f"Respecte les contraintes dures : {not gplan.detecte_contrainte(resultat_tabou)}")

    return resultat_tabou, scores[-1]

def solve_multi(Ns, Ds, preferences_eqs, reductions_eqs, attributs_eqs, eqs_to_global, global_to_eqs, planning_initiaux=None, jours_a_modifier=None, skip_optims=None):
    E = len(Ns) # nombre d'équipes

    preferences_eqs = copy.deepcopy(preferences_eqs)

    resultat_eqs = []
    scores_eqs = []

    # Première boucle : mettre à jour les préférences pour les équipes avec skip_optim=True
    for eq in range(E):
        if not skip_optims[eq]:
            continue

        resultat_eq = planning_initiaux[eq]  # pas besoin d'appeler solve_mono ici
        
        # modification des preferences de toutes les autres equipes
        for eqb in range(E):
            resultat_eq_global = [eqs_to_global[eq][mdc] for mdc in resultat_eq]
            resultat_eq_eqb = [global_to_eqs[eqb][mdc] if mdc in global_to_eqs[eqb] else -1 for mdc in resultat_eq_global]

            for jour, (mdc_garde, mdc_astreinte) in enumerate(zip(resultat_eq_eqb[:Ds[eq]], resultat_eq_eqb[Ds[eq]:])):
                if jour >= Ds[eqb]:
                    break
                if mdc_garde != -1:
                    preferences_eqs[eqb][mdc_garde, jour] = NEG_PREF_TEAM
                if mdc_garde != -1 and jour+1 < Ds[eqb]:
                    preferences_eqs[eqb][mdc_garde, jour+1] = NEG_PREF_TEAM
                if mdc_garde != -1 and jour-1 >= 0:
                    preferences_eqs[eqb][mdc_garde, jour-1] = min(-5, preferences_eqs[eqb][mdc_garde, jour-1])
                if mdc_astreinte != -1 and jour-1 >= 0:
                    preferences_eqs[eqb][mdc_astreinte, jour-1] = min(-5, preferences_eqs[eqb][mdc_astreinte, jour-1])
                if mdc_astreinte != -1:
                    preferences_eqs[eqb][mdc_astreinte, jour] = NEG_PREF_TEAM

    for eq in range(E):
        # résolution planning eq
        resultat_eq, score_final_eq = solve_mono(Ds[eq], Ns[eq], preferences_eqs[eq], reductions_eqs[eq], attributs_eqs[eq], eq=eq+1, planning_initial=planning_initiaux[eq] if planning_initiaux else None, jours_gras=jours_a_modifier[eq] if jours_a_modifier else None, skip_optim=skip_optims[eq] if skip_optims else False)
        resultat_eqs.append(resultat_eq)
        scores_eqs.append(score_final_eq)

        # modification des preferences de toutes les autres equipes
        for eqb in range(E):
            resultat_eq_global = [eqs_to_global[eq][mdc] for mdc in resultat_eq]
            resultat_eq_eqb = [global_to_eqs[eqb][mdc] if mdc in global_to_eqs[eqb] else -1 for mdc in resultat_eq_global]

            for jour, (mdc_garde, mdc_astreinte) in enumerate(zip(resultat_eq_eqb[:Ds[eq]], resultat_eq_eqb[Ds[eq]:])):
                if jour >= Ds[eqb]:
                    break
                if mdc_garde != -1:
                    preferences_eqs[eqb][mdc_garde, jour] = NEG_PREF_TEAM
                if mdc_garde != -1 and jour+1 < Ds[eqb]:
                    preferences_eqs[eqb][mdc_garde, jour+1] = NEG_PREF_TEAM
                if mdc_garde != -1 and jour-1 >= 0:
                    preferences_eqs[eqb][mdc_garde, jour-1] = min(-5, preferences_eqs[eqb][mdc_garde, jour-1])
                if mdc_astreinte != -1 and jour-1 >= 0:
                    preferences_eqs[eqb][mdc_astreinte, jour-1] = min(-5, preferences_eqs[eqb][mdc_astreinte, jour-1])
                if mdc_astreinte != -1:
                    preferences_eqs[eqb][mdc_astreinte, jour] = NEG_PREF_TEAM
    
    return resultat_eqs, scores_eqs
