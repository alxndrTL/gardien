import numpy as np
import tqdm

from config import MIN_P_AC, HEURISTIC_NEG_PREF_AC

"""
ACO

Parcours de graphe, en affectant les médecins aux gardes en fonction des niveaux 
de phéromones et des heuristiques (ici, leur préférence) en respectant les contraintes.

-matrice de phéromones τ[m][t][s] : l'attractivité d'affecter le médecin m au shift s (0 pour garde, 1 pour astreinte) le jour t.
-heuristique η[m][t][s] : préférences du médecin m au shift s (0 ou 1) du jour t

probabilité d'affecter le médecin m à une garde est proportionnelle à (τ[m][t][s])^alpha * (η[m][t][s])^beta

On choisit rho=0.1. on s'est rendu compte qu'en le prenant trop petit, 
les bonnes solutions n'étaient pas re-visitées et donc les performances finales n'étaient pas très bonnes
avec un rho plus grand, l'algo est lent mais plus stable
alpha=0.1, beta=2 : trouvés empiriquement
"""

def recherche_ant_colony(num_ants, num_iterations, alpha, beta, rho, gplan, eq=None, sol_initiale=None, jours_gras=None):
    """
    Paramètres:
    - num_ants: nombre de fourmis
    - num_iterations: nombre d'itérations
    - alpha: facteur sur les phéromones (cf commentaire au-dessus)
    - beta: facteur sur l'heuristique
    - rho: taux d'évaporation des phéromones
    - gplan: GestionnairePlanning.
    - eq: numéro de l'équipe (seulement pour l'affichage)
    - sol_initiale: possible solution de départ
    - jours_gras: jours à modifier
    """

    N = gplan.N  # Number of doctors
    D = gplan.D  # Number of days
    
    # init les phéromones τ[m][t][s]
    tau0 = 1.0
    pheromone = np.ones((N, D, 2)) * tau0  # 2 for garde and astreinte

    # si on a une solution initiale, on double le niveau de phéromone sur ses chemins
    if sol_initiale is not None:
        planning_gardes = sol_initiale[:D]
        planning_astreintes = sol_initiale[D:]
        for t in range(D):
            if planning_gardes[t] != -1:
                pheromone[planning_gardes[t], t, 0] *= 2
            if planning_astreintes[t] != -1:
                pheromone[planning_astreintes[t], t, 1] *= 2

    # init des heuristiques η[m][t][s]
    heuristic_garde = np.zeros((N, D))
    for i in range(N):
        for t in range(D):
            pref = gplan.preferences[i][t]
            if pref >= 0:
                heuristic_garde[i][t] = pref + 1
            else:
                heuristic_garde[i][t] = HEURISTIC_NEG_PREF_AC # environ 0, petite valeure positive (pour éviter les NaN...)

    # pour les jours en gras, on réduit l'heuristique pour les médecins actuellement affectés
    if sol_initiale is not None and jours_gras is not None:
        for t in jours_gras['garde']:
            if sol_initiale[t] != -1:
                heuristic_garde[sol_initiale[t], t] *= HEURISTIC_NEG_PREF_AC # environ 0, petite valeure positive (pour éviter les NaN...)
        for t in jours_gras['astreinte']:
            if sol_initiale[D + t] != -1:
                heuristic_garde[sol_initiale[D + t], t] *= HEURISTIC_NEG_PREF_AC # environ 0

    # pour les astreintes, on ne prend pas en compte les préférences
    heuristic_astreinte = np.ones((N, D))

    heuristic = np.stack((heuristic_garde, heuristic_astreinte), axis=2)

    best_planning = None
    best_score = float('inf')
    scores = []

    pbar = tqdm.tqdm(range(num_iterations))

    # lancement de la recherche
    for _ in pbar:
        all_solutions = []

        # chaque fourmi=une solution
        for _ in range(num_ants):
            planning = construct_solution(pheromone, heuristic, gplan, alpha, beta, sol_initiale, jours_gras)
            planning = gplan.forcer_contrainte(planning)
            score = gplan.calcule_critere(planning)
            all_solutions.append((planning, score))
            if score < best_score:
                best_planning = planning.copy()
                best_score = score

        # calcul des phéromones
        pheromone = pheromone * (1 - rho)  # Evaporation

        # ici, on récupère la meilleure fourmi, et on va augmenter les phéronomones sur son chemin
        best_ant_solution, best_ant_score = min(all_solutions, key=lambda x: x[1])
        delta_tau = 1.0 / best_ant_score # plus le score est grand, plus on va déposer de phéromone (cf papier 4.3.4)

        # maj des phéromones
        for t in range(D):
            i_garde = best_ant_solution[t]
            pheromone[i_garde, t, 0] += delta_tau
            i_astreinte = best_ant_solution[D + t]
            pheromone[i_astreinte, t, 1] += delta_tau

        scores.append(best_score)
        pbar.set_description(f"\033[1m\033[35m[GARDIEN]\033[0m [\033[34mÉquipe {eq}\033[0m \033[1m\033[35m1/2\033[0m] \033[32mmeilleur score: \033[1m{best_score:.0f}\033[0m")
    pbar.close()
    
    return best_planning, best_score, scores

def construct_solution(pheromone, heuristic, gplan, alpha, beta, sol_initiale=None, jours_gras=None):
    """
    fonction annexe qui construit une solution à partir des phéronomones et des heuristiques
    """

    N = gplan.N
    D = gplan.D
    planning_gardes = np.zeros(D, dtype=int)
    planning_astreintes = np.zeros(D, dtype=int)

    # si on a une solution initiale, on part de celle-ci
    if sol_initiale is not None:
        planning_gardes = sol_initiale[:D].copy()
        planning_astreintes = sol_initiale[D:].copy()

    for t in range(D):
        if sol_initiale is not None:
            if planning_gardes[t] != -1 and (jours_gras is None or t not in jours_gras['garde']):
                continue # si jour déjà assigné et à ne pas modifier, on skip

        unavailable_doctors_garde = set()
        if t > 0:
            unavailable_doctors_garde.add(planning_gardes[t-1])
        
        # si c'est un jour en gras, on évite le médecin actuellement assigné
        if jours_gras is not None and t in jours_gras['garde'] and sol_initiale is not None:
            unavailable_doctors_garde.add(sol_initiale[t])

        available_doctors_garde = set(range(N)) - unavailable_doctors_garde

        # calcul des probabilités pour les médecins disponibles
        probs_garde = []
        doctors_garde = []
        for i in available_doctors_garde:
            tau = np.maximum(pheromone[i, t, 0], MIN_P_AC)
            eta = np.maximum(heuristic[i, t, 0], MIN_P_AC)
            p = np.exp(alpha * np.log(tau) + beta * np.log(eta))
            probs_garde.append(p)
            doctors_garde.append(i)

        probs_garde = np.array(probs_garde)
        probs_garde = np.maximum(probs_garde, MIN_P_AC)
        probs_garde = np.nan_to_num(probs_garde, nan=MIN_P_AC)
        if probs_garde.sum() == 0:
            probs_garde = np.ones_like(probs_garde)
        probs_garde /= probs_garde.sum()

        selected_doctor_garde = np.random.choice(doctors_garde, p=probs_garde)
        planning_gardes[t] = selected_doctor_garde

        # si l'astreinte est déjà assignée et n'est pas à modifier, on continue
        if sol_initiale is not None:
            if planning_astreintes[t] != -1 and (jours_gras is None or t not in jours_gras['astreinte']):
                continue

        unavailable_doctors_astreinte = {planning_gardes[t]}
        if t > 0:
            unavailable_doctors_astreinte.add(planning_gardes[t-1])
        
        # si c'est un jour en gras, on évite le médecin actuellement assigné
        if jours_gras is not None and t in jours_gras['astreinte'] and sol_initiale is not None:
            unavailable_doctors_astreinte.add(sol_initiale[D + t])

        available_doctors_astreinte = set(range(N)) - unavailable_doctors_astreinte

        probs_astreinte = []
        doctors_astreinte = []
        for i in available_doctors_astreinte:
            tau = np.maximum(pheromone[i, t, 1], MIN_P_AC)
            eta = np.maximum(heuristic[i, t, 1], MIN_P_AC)
            p = np.exp(alpha * np.log(tau) + beta * np.log(eta)) # cf section 4.3.4 papier
            probs_astreinte.append(p)
            doctors_astreinte.append(i)

        probs_astreinte = np.array(probs_astreinte)
        probs_astreinte = np.maximum(probs_astreinte, MIN_P_AC)
        probs_astreinte = np.nan_to_num(probs_astreinte, nan=MIN_P_AC)
        if probs_astreinte.sum() == 0:
            probs_astreinte = np.ones_like(probs_astreinte)
        probs_astreinte /= probs_astreinte.sum()

        selected_doctor_astreinte = np.random.choice(doctors_astreinte, p=probs_astreinte)
        planning_astreintes[t] = selected_doctor_astreinte

    return np.concatenate((planning_gardes, planning_astreintes))
