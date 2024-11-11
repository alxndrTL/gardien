import numpy as np
from collections import deque # utilisé pour la file tabou
import tqdm
import time

from definition import GestionnairePlanning
from config import TENTATIVE_MULT_T

"""
Plusieurs choix ont été fait:

-pour le voisinage, on choisit de changer une garde ou une astreinte (on change de praticien une garde ou astreinte aléatoire)
(idem que pour recuit simule)

-pour la liste tabou, on utilise une file (FIFO) de taille maximale max_file (paramètre à choisir).
on lui donne une taille maximale. on choisit d'y stocker les solutions et non les mouvements*

*chaque solution représente 6mois=30x6=180jours de garde, donc 180 entiers.
Chacun étant codé sur 32 bits donc 4 octets, une liste "pèse" donc en mémoire <1Ko.
Pour des tailles de liste tabou <1000, on a donc une empreinte mémoire de 1Mo
(10 000 fois moins que la capacité d'ordinateurs grands publics)

Note: comme on manipule des vecteurs numpy (les solutions/plannings), on fait des copies
pour éviter de mauvaises surprises.
"""

def planning_voisin(planning, gplan, jours_gras=None):
    """
    Fonction qui retourne une solution voisine en modifiant soit une garde soit une astreinte.
    Si jours_gras est fourni, évite de réutiliser les médecins des jours à modifier.
    """
    voisin = planning.copy()
    D = gplan.D
    
    # on modifie une garde (0) ou une astreinte (1) ?
    type_modif = np.random.randint(0, 2)
    # quel jour on va modifier ?
    jour = np.random.randint(0, D)
    # calcule de l'index
    index = jour if type_modif == 0 else jour + D # D premiers jours=gardes, les D suivants=astreintes
    # quels médecins sont disponibles (éviter le médecin actuel si jour en gras)
    medecins_disponibles = list(range(gplan.N))
    if jours_gras and ((type_modif == 0 and jour in jours_gras['garde']) or 
                      (type_modif == 1 and jour in jours_gras['astreinte'])):
        medecins_disponibles.remove(voisin[index])
    # choix au hasard
    voisin[index] = np.random.choice(medecins_disponibles)
    return voisin

def recherche_tabou(num_iters, num_voisins, max_stagnation, len_tabou, gplan: GestionnairePlanning, sol=None, max_dist=None, planning_initial=None, jours_gras=None, eq=None):
    """
    Recherche un planning qui minimise le critère défini dans definition.py par méthode tabou.
    L'algorithme arrête sa recherche lorsqu'il "stagne": aucune amélioration sur max_stagnation étapes successives.
    Renvoie le meilleur individu trouvé pendant toute la recherche, et la liste des critères obtenus au fil de la recherche.

    Paramètres:
    -sol: permet de faire partir la recherche à partir d'une solution donnée
    -max_dist: permet de limiter la recherche de plannings à max_dist du planning initial
    -planning_initial: couplé à max_dist, permet de limiter la recherche en terme de distance
    -jours_gras: liste des jours qu'il faut modifier (utiliser dans planning_voisin)
    -eq: numéro de l'équipe (seulement utilisé pour l'affichage)
    """

    tabou = deque(maxlen=len_tabou)

    if sol is None: # si pas de sol initiale donnée, on en génère une au hasard
        sol = gplan.solution_initiale()
    sol_critere = gplan.calcule_critere(sol)

    meilleur_sol = sol.copy()
    meilleur_critere = sol_critere # pour critère d'aspiration
    stagnation = 0
    scores = []

    pbar = tqdm.tqdm(range(num_iters))

    for _ in range(num_iters):
        voisin_critere_min = float('inf')
        meilleur_voisin = None

        tentatives = 0
        max_tentatives = TENTATIVE_MULT_T*num_voisins
        compteur = 0

        # cette boucle s'occupe de générer num_voisins
        # il se peut qu'elle soit très lente car on limite la recherche en distance,
        # donc on limite à 3*num_voisins (3=TENTATIVE_MULT_T)
        while tentatives < max_tentatives and compteur < num_voisins:
            voisin = planning_voisin(sol, gplan, jours_gras)
            voisin = gplan.forcer_contrainte(voisin)
            tentatives += 1

            if planning_initial is not None and max_dist is not None:
                masque = planning_initial != -1
                dist = np.sum(voisin[masque] != planning_initial[masque])
                if dist > max_dist: # distance dépassée, on ignore le voisin trouvé
                    continue
            
            compteur += 1

            voisin_critere = gplan.calcule_critere(voisin)

            # critère d'aspiration : A(f(s)) prend la valeur de la meilleure solution s*
            if voisin.tobytes() not in tabou or voisin_critere < meilleur_critere:
                if voisin_critere < voisin_critere_min:
                    voisin_critere_min = voisin_critere
                    meilleur_voisin = voisin.copy()
        
        # si on n'a trouvé aucun voisin valide, on saute l'itération
        if meilleur_voisin is None:
            stagnation += 1
            scores.append(sol_critere)
            continue
        
        # ajout à la liste tabou
        tabou.append(sol.tobytes())
        sol = meilleur_voisin
        sol_critere = voisin_critere_min

        # utilisé pour tracker la stagnation + la meilleure solution trouvée depuis le début (qu'on va renvoyer)
        if sol_critere < meilleur_critere:
            meilleur_sol = sol.copy()
            meilleur_critere = sol_critere
            stagnation = 0
        else:
            stagnation += 1

        scores.append(voisin_critere_min)

        pbar.set_description(f"\033[1m\033[35m[GARDIEN]\033[0m [\033[34mÉquipe {eq}\033[0m \033[1m\033[35m2/2\033[0m] \033[32mmeilleur score: \033[1m{sol_critere:.0f}\033[0m")
        pbar.update(1)

        if stagnation >= max_stagnation:
            #print(f"Arrêt après {_+1} itérations dû à la stagnation.")
            break

    remaining = num_iters - pbar.n
    for _ in range(remaining):
        time.sleep(0.005)
        pbar.update(1)
    pbar.close()

    return meilleur_sol, scores
