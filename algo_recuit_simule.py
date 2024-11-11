import numpy as np

from definition import GestionnairePlanning

"""
[NON UTILISE PAR GARDIEN]

Plusieurs choix ont été fait:

-pour le voisinage, on choisit de changer une garde ou une astreinte (on change de praticien une garde ou astreinte aléatoire)
(idem que pour recherche tabou)

-l'implémentation générale n'a pas nécessité de modification par rapport à la trame proposée dans le cours.

-la majorité du travail pour cet algorithme a été de comprendre l'influence des paramètres
et de trouver les bons pour notre problème (T_0 et a notamment)

Note: comme on manipule des vecteurs numpy (les solutions/plannings), on fait des copies
pour éviter de mauvaises surprises.
"""

def planning_voisin(planning, gplan):
    """
    fonction qui retourne une solution voisine
    (on échange juste le médecin d'une garde aléatoire)
    """
    voisin = planning.copy()
    creneau_a_modifier = np.random.randint(0, len(planning))
    nouveau_mdc = gplan.random_mdc()
    voisin[creneau_a_modifier] = nouveau_mdc
    return voisin

def recherche_recuit_simule(nb_iters_cycle, T_0, a, gplan: GestionnairePlanning, sol=None):
    """
    Recherche un planning qui minimise le critère défini dans definition.py par recuit simulé.
    Renvoie le meilleur individu trouvé pendant toute la recherche, et la liste des critères obtenus au fil de la recherche.
    """
    
    # si une sol initiale est passée, on la prend. sinon on la génère aléatoirement
    if sol is None:
        sol = gplan.solution_initiale()
    else:
        sol_ref = sol.copy()
    sol_critere = gplan.calcule_critere(sol)

    meilleur_sol = sol.copy()
    meilleur_critere = sol_critere

    k = 0
    nouveau_cycle = True
    T = T_0
    scores = []

    # on fait plusieurs cycles de température (T_0->0) tant qu'on trouve des amélioration
    # dans chaque cycle, on fait nb_iters_cycle
    while nouveau_cycle:
        nb_iter = 0
        nouveau_cycle = False

        while nb_iter < nb_iters_cycle:
            k += 1
            nb_iter += 1

            # génération d'un voisin
            voisin = planning_voisin(sol, gplan)
            voisin = gplan.forcer_contrainte(voisin)
            voisin_critere = gplan.calcule_critere(voisin)

            # différence de critère entre le voisin et notre sol actuelle
            df = voisin_critere - sol_critere

            # le voisin est meilleur: on l'accepte
            if df < 0:
                sol = voisin
                sol_critere = voisin_critere
                nouveau_cycle = True
            else: # sinon, on l'accepte mais avec une probabilité (qui dépend de df et T)
                prob = np.exp(-df/T)
                q = np.random.uniform()

                if q < prob:
                    sol = voisin
                    sol_critere = voisin_critere
                    nouveau_cycle = True
            
            if sol_critere < meilleur_critere:
                meilleur_sol = sol.copy()
                meilleur_critere = sol_critere
        
        scores.append(meilleur_critere)
        
        T = a * T # on refroidie / baisse la température

    return meilleur_sol, scores
