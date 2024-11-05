import numpy as np

from definition import GestionnairePlanning

"""
Plusieurs choix ont été fait:

-pour le voisinage, on choisit de changer une garde ou une astreinte (on change de praticien une garde ou astreinte aléatoire)
(idem que pour recherche tabou)

-l'implémentation générale n'a pas nécessité de modification par rapport à la trame proposée dans le cours.

-la majorité du travail pour cet algorithme a été de comprendre l'influence des paramètres
et de trouver les bons pour notre problème (T_0 et a notamment)

Note: comme on manipule des vecteurs numpy (les solutions/plannings), on fait des copies
pour éviter de mauvaises surprises.
"""

def planning_voisin(planning, gplan, creneau_a_modifier=None):
    """
    fonction qui retourne une solution voisine
    (on échange juste le médecin d'une garde aléatoire)
    """
    voisin = planning.copy()
    if creneau_a_modifier is None:
        creneau_a_modifier = np.random.randint(0, len(planning))
    nouveau_mdc = gplan.random_mdc()
    voisin[creneau_a_modifier] = nouveau_mdc
    return voisin

def recherche_recuit_simule(nb_iters_cycle, T_0, a, gplan: GestionnairePlanning, sol=None, max_dist=None, changer_jour=None):
    """
    Recherche un planning qui minimise le critère défini dans definition.py par recuit simulé.
    Renvoie le meilleur individu trouvé pendant toute la recherche, et la liste des critères obtenus au fil de la recherche.
    """

    assert not(max_dist is not None and sol is None), "une solution initiale doit être donnée pour que max_dist ait un sens"
    assert not(changer_jour is not None and sol is None), "une solution initiale doit être donnée pour que changer_jour ait un sens"
    assert not(changer_jour is not None and max_dist is None), "une distance max doit être donnée pour que changer_jour ait un sens"

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

    while nouveau_cycle:
        nb_iter = 0
        nouveau_cycle = False

        while nb_iter < nb_iters_cycle:
            k += 1
            nb_iter += 1

            if max_dist:
                dist = max_dist+1
                jour_change = False
                while dist > max_dist and (not jour_change):
                    voisin = planning_voisin(sol, gplan, changer_jour)
                    voisin = gplan.forcer_contrainte(voisin)
                    dist = gplan.distance_sol(voisin, sol_ref)
                    if changer_jour is not None:
                        jour_change = not(voisin[changer_jour] == sol_ref[changer_jour])
                    else:
                        jour_change = True
            else:
                voisin = planning_voisin(sol, gplan)
                voisin = gplan.forcer_contrainte(voisin)
            voisin_critere = gplan.calcule_critere(voisin)

            df = voisin_critere - sol_critere

            if df < 0:
                sol = voisin
                sol_critere = voisin_critere
                nouveau_cycle = True
            else:
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
        
        T = a * T

    return meilleur_sol, scores
