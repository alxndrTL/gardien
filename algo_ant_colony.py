import numpy as np
import tqdm

from config import MIN_P_AC, HEURISTIC_NEG_PREF_AC

"""
Ant Colony Optimization for medical staff scheduling.

The algorithm constructs solutions by simulating the behavior of ants,
which probabilistically assign doctors to shifts based on pheromone levels
and heuristic information, while respecting constraints.

Key components:
- Pheromone matrix τ[i][t][s]: Represents the desirability of assigning doctor i
  to shift s (0 for garde, 1 for astreinte) on day t.
- Heuristic information η[i][t][s]: Based on doctors' preferences.
- Probability of assigning a doctor to a shift is proportional to
  (τ[i][t][s])^α * (η[i][t][s])^β.

Constraints are enforced during the construction of solutions.
"""

def recherche_ant_colony(num_ants, num_iterations, alpha, beta, rho, gplan, eq=None, sol_initiale=None, jours_gras=None):
    """
    Ant Colony Optimization algorithm to find the best planning.
    
    Parameters:
    - num_ants: Number of ants in each iteration.
    - num_iterations: Maximum number of iterations.
    - alpha: Pheromone importance factor.
    - beta: Heuristic information importance factor.
    - rho: Pheromone evaporation rate.
    - gplan: An instance of GestionnairePlanning.

    Returns:
    - best_planning: The best planning found.
    - best_score: The criterion value of the best planning.
    - scores: List of best scores from each iteration.
    """
    N = gplan.N  # Number of doctors
    D = gplan.D  # Number of days
    
    # Initialize pheromone levels τ[i][t][s]
    tau0 = 1.0
    pheromone = np.ones((N, D, 2)) * tau0  # 2 for garde and astreinte

    # Si on a une solution initiale, on augmente le niveau de phéromone sur ses chemins
    if sol_initiale is not None:
        planning_gardes = sol_initiale[:D]
        planning_astreintes = sol_initiale[D:]
        for t in range(D):
            if planning_gardes[t] != -1:  # Si le jour a un médecin assigné
                pheromone[planning_gardes[t], t, 0] *= 2
            if planning_astreintes[t] != -1:  # Si le jour a un médecin assigné
                pheromone[planning_astreintes[t], t, 1] *= 2

    # Initialize heuristic information η[i][t][s]
    # For guardes
    heuristic_garde = np.zeros((N, D))
    for i in range(N):
        for t in range(D):
            pref = gplan.preferences[i][t]
            if pref >= 0:
                heuristic_garde[i][t] = pref + 1  # Shift to make it positive
            else:
                heuristic_garde[i][t] = HEURISTIC_NEG_PREF_AC  # Small positive value for negative preferences

    # Pour les jours en gras, on réduit l'heuristique pour les médecins actuellement affectés
    if sol_initiale is not None and jours_gras is not None:
        for t in jours_gras['garde']:
            if sol_initiale[t] != -1:
                heuristic_garde[sol_initiale[t], t] *= HEURISTIC_NEG_PREF_AC
        for t in jours_gras['astreinte']:
            if sol_initiale[D + t] != -1:
                heuristic_garde[sol_initiale[D + t], t] *= HEURISTIC_NEG_PREF_AC

    # For astreintes, set heuristic values to 1 (or adjust as needed)
    heuristic_astreinte = np.ones((N, D))

    heuristic = np.stack((heuristic_garde, heuristic_astreinte), axis=2)  # Shape (N, D, 2)

    best_planning = None
    best_score = float('inf')
    scores = []

    pbar = tqdm.tqdm(range(num_iterations))

    for _ in pbar:
        all_solutions = []
        for _ in range(num_ants):
            planning = construct_solution(pheromone, heuristic, gplan, alpha, beta, sol_initiale, jours_gras)
            planning = gplan.forcer_contrainte(planning)
            score = gplan.calcule_critere(planning)
            all_solutions.append((planning, score))
            if score < best_score:
                best_planning = planning.copy()
                best_score = score
        # Update pheromones
        pheromone = pheromone * (1 - rho)  # Evaporation

        # Deposit pheromone based on the best ant's solution in this iteration
        # You can also use other strategies like depositing pheromone from multiple ants
        best_ant_solution, best_ant_score = min(all_solutions, key=lambda x: x[1])

        delta_tau = 1.0 / best_ant_score  # Amount of pheromone to deposit

        # Update pheromones based on the best ant's solution
        for t in range(D):
            # Update for garde
            i_garde = best_ant_solution[t]
            pheromone[i_garde, t, 0] += delta_tau
            # Update for astreinte
            i_astreinte = best_ant_solution[D + t]
            pheromone[i_astreinte, t, 1] += delta_tau

        scores.append(best_score)
        #print(f"Iteration {iteration + 1}/{num_iterations}, Best Score: {best_score}")
        pbar.set_description(f"\033[1m\033[35m[GARDIEN]\033[0m [\033[34mÉquipe {eq}\033[0m \033[1m\033[35m1/2\033[0m] \033[32mmeilleur score: \033[1m{best_score:.0f}\033[0m")
    pbar.close()
    
    return best_planning, best_score, scores

def construct_solution(pheromone, heuristic, gplan, alpha, beta, sol_initiale=None, jours_gras=None):
    N = gplan.N
    D = gplan.D
    planning_gardes = np.zeros(D, dtype=int)
    planning_astreintes = np.zeros(D, dtype=int)

    # Si on a une solution initiale, on part de celle-ci
    if sol_initiale is not None:
        planning_gardes = sol_initiale[:D].copy()
        planning_astreintes = sol_initiale[D:].copy()

    for t in range(D):
        # Si le jour est déjà assigné et n'est pas à modifier, on continue
        if sol_initiale is not None:
            if planning_gardes[t] != -1 and (jours_gras is None or t not in jours_gras['garde']):
                continue

        # Assign Garde
        unavailable_doctors_garde = set()
        if t > 0:
            unavailable_doctors_garde.add(planning_gardes[t-1])
        
        # Si c'est un jour en gras, on évite le médecin actuellement assigné
        if jours_gras is not None and t in jours_gras['garde'] and sol_initiale is not None:
            unavailable_doctors_garde.add(sol_initiale[t])

        available_doctors_garde = set(range(N)) - unavailable_doctors_garde

        # Calcul des probabilités pour les médecins disponibles
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

        # Si l'astreinte est déjà assignée et n'est pas à modifier, on continue
        if sol_initiale is not None:
            if planning_astreintes[t] != -1 and (jours_gras is None or t not in jours_gras['astreinte']):
                continue

        # Assign Astreinte
        unavailable_doctors_astreinte = {planning_gardes[t]}
        if t > 0:
            unavailable_doctors_astreinte.add(planning_gardes[t-1])
        
        # Si c'est un jour en gras, on évite le médecin actuellement assigné
        if jours_gras is not None and t in jours_gras['astreinte'] and sol_initiale is not None:
            unavailable_doctors_astreinte.add(sol_initiale[D + t])

        available_doctors_astreinte = set(range(N)) - unavailable_doctors_astreinte

        probs_astreinte = []
        doctors_astreinte = []
        for i in available_doctors_astreinte:
            tau = np.maximum(pheromone[i, t, 1], MIN_P_AC)
            eta = np.maximum(heuristic[i, t, 1], MIN_P_AC)
            p = np.exp(alpha * np.log(tau) + beta * np.log(eta))
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
