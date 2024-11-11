import copy
import numpy as np

from definition import GestionnairePlanning
from algo_ant_colony import recherche_ant_colony
from algo_tabou import recherche_tabou
from config import *

def solve_mono(nombre_jours, nombre_mdc, preferences, reductions=None, attributs=None, implications=None, eq=None, planning_initial=None, jours_gras=None, jours_soulignes=None, skip_optim=False):
    """
    Optimise un seul planning avec ACO+TS
    """

    if skip_optim and planning_initial is not None:
        return np.array(planning_initial), 0
    
    gplan = GestionnairePlanning(nombre_mdc, nombre_jours, preferences, reductions, attributs, implications, jours_gras, jours_soulignes, planning_initial)

    if planning_initial is not None:
        planning_initial = np.array(planning_initial)
        nb_vides = np.sum(np.array(planning_initial) == -1)
        
        nb_gras = len(jours_gras['garde']) + len(jours_gras['astreinte'])
        max_dist = MAX_DIST + nb_gras + nb_vides # on rajoute à max_dist les jours en gras et les cases vides (non comptées dans la distance)
    else:
        planning_initial = None
        max_dist = None

    # PREMIERE ETAPE : ANT COLONY OPTIMIZATION (ACO)
    resultat_aoc, _, _ = recherche_ant_colony(NUM_ANTS, NUM_ITERS_AC, ALPHA, BETA, RHO, gplan, eq=eq, sol_initiale=planning_initial, jours_gras=jours_gras)

    # DEUXIEME ETAPE : TABOU SEARCH (TS)
    resultat_tabou, scores = recherche_tabou(NUM_ITERS_T, NUM_VOISINS, MAX_STAGNATION, LEN_TABOU, gplan, sol=resultat_aoc, eq=eq, max_dist=max_dist, planning_initial=planning_initial, jours_gras=jours_gras)

    return resultat_tabou, scores[-1]

def solve_multi(Ns, Ds, preferences_eqs, reductions_eqs, attributs_eqs, implications_eqs, eqs_to_global, global_to_eqs, planning_initiaux=None, jours_a_modifier=None, jours_fixes=None, skip_optims=None):
    """
    Optimise plusieurs plannings séquentiellement.
    Optimise d'abord le premier planning, puis modifie les préférences des autres plannings pour empêcher les collisions.
    Modifie ensuite le second, modifie les préférences etc. Cela empêche les collisions.
    """

    E = len(Ns) # nombre d'équipes

    preferences_eqs = copy.deepcopy(preferences_eqs)

    resultat_eqs = []
    scores_eqs = []

    # ici, on ne fait pas encore d'optimisation
    # on boucle sur les équipes, et si jamais certaines ont des plannings déjà pleins (avec skip_optim)
    # on va modifie les préférences des autres équipes pour empêcher les autres plannings d'employer des mdc déjà employés
    for eq in range(E):
        if not skip_optims[eq]:
            continue

        resultat_eq = planning_initiaux[eq]
        
        # modification des preferences de toutes les autres equipes
        for eqb in range(E):
            resultat_eq_global = [eqs_to_global[eq][mdc] for mdc in resultat_eq]
            resultat_eq_eqb = [global_to_eqs[eqb][mdc] if mdc in global_to_eqs[eqb] else -1 for mdc in resultat_eq_global]

            for jour, (mdc_garde, mdc_astreinte) in enumerate(zip(resultat_eq_eqb[:Ds[eq]], resultat_eq_eqb[Ds[eq]:])):
                if jour >= Ds[eqb]:
                    break
                if mdc_garde != -1:
                    preferences_eqs[eqb][mdc_garde, jour] = NEG_PREF_TEAM # on empeche l'assignation le même jour d'une garde
                if mdc_garde != -1 and jour+1 < Ds[eqb]:
                    preferences_eqs[eqb][mdc_garde, jour+1] = NEG_PREF_TEAM # on empeche l'assignation de lendemain d'une garde
                if mdc_garde != -1 and jour-1 >= 0:
                    preferences_eqs[eqb][mdc_garde, jour-1] = min(SEUIL_PREF_NEG_ASTREINTE, preferences_eqs[eqb][mdc_garde, jour-1]) # on empêche l'assignation d'une garde le jour d'avant une garde
                if mdc_astreinte != -1 and jour-1 >= 0:
                    preferences_eqs[eqb][mdc_astreinte, jour-1] = min(SEUIL_PREF_NEG_ASTREINTE, preferences_eqs[eqb][mdc_astreinte, jour-1]) # on empêche l'assignation d'une garde le jour d'avant une astreinte
                if mdc_astreinte != -1:
                    preferences_eqs[eqb][mdc_astreinte, jour] = NEG_PREF_TEAM # on empche l'assignation le même jour d'une astreinte

                # note : SEUIL_PREF_NEG_ASTREINTE (-5 de base) est le seuil de préférence en dessous duquel on n'affecte pas d'astreinte (strictement)
                # donc si la préférence est -5, on peut affecter de astreintes, si -6 non.
                # donc lorsqu'on place à SEUIL_PREF_NEG_ASTREINTE un jour, on empêche l'assignation d'une garde mais pas d'une astreinte

    # deuxième boucle : optimisation de chaque planning, séquentiellement
    for eq in range(E):
        # résolution planning eq
        resultat_eq, score_final_eq = solve_mono(Ds[eq], Ns[eq], preferences_eqs[eq], reductions_eqs[eq], attributs_eqs[eq], implications_eqs[eq], eq=eq+1, planning_initial=planning_initiaux[eq] if planning_initiaux else None, jours_gras=jours_a_modifier[eq] if jours_a_modifier else None, jours_soulignes=jours_fixes[eq] if jours_fixes else None, skip_optim=skip_optims[eq] if skip_optims else False)
        resultat_eqs.append(resultat_eq)
        scores_eqs.append(score_final_eq)

        # modification des preferences de toutes les autres equipes (cf boucle d'avant, c'est le même code)
        for eqb in range(E):
            resultat_eq_global = [eqs_to_global[eq][mdc] if mdc != -1 else -1 for mdc in resultat_eq]
            resultat_eq_eqb = [global_to_eqs[eqb][mdc] if mdc in global_to_eqs[eqb] else -1 for mdc in resultat_eq_global]

            for jour, (mdc_garde, mdc_astreinte) in enumerate(zip(resultat_eq_eqb[:Ds[eq]], resultat_eq_eqb[Ds[eq]:])):
                if jour >= Ds[eqb]:
                    break
                if mdc_garde != -1:
                    preferences_eqs[eqb][mdc_garde, jour] = NEG_PREF_TEAM
                if mdc_garde != -1 and jour+1 < Ds[eqb]:
                    preferences_eqs[eqb][mdc_garde, jour+1] = NEG_PREF_TEAM
                if mdc_garde != -1 and jour-1 >= 0:
                    preferences_eqs[eqb][mdc_garde, jour-1] = min(SEUIL_PREF_NEG_ASTREINTE, preferences_eqs[eqb][mdc_garde, jour-1])
                if mdc_astreinte != -1 and jour-1 >= 0:
                    preferences_eqs[eqb][mdc_astreinte, jour-1] = min(SEUIL_PREF_NEG_ASTREINTE, preferences_eqs[eqb][mdc_astreinte, jour-1])
                if mdc_astreinte != -1:
                    preferences_eqs[eqb][mdc_astreinte, jour] = NEG_PREF_TEAM
    
    return resultat_eqs, scores_eqs
