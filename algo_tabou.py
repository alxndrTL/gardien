import random
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
    
    # Choisir si on modifie une garde (0) ou une astreinte (1)
    type_modif = np.random.randint(0, 2)
    
    # Sélectionner un jour au hasard
    jour = np.random.randint(0, D)
    
    # Calculer l'index dans le planning en fonction du type de modification
    index = jour if type_modif == 0 else jour + D
    
    # Générer la liste des médecins disponibles (éviter le médecin actuel si jour en gras)
    medecins_disponibles = list(range(gplan.N))
    if jours_gras and ((type_modif == 0 and jour in jours_gras['garde']) or 
                      (type_modif == 1 and jour in jours_gras['astreinte'])):
        medecins_disponibles.remove(voisin[index])
    
    # Choisir un nouveau médecin au hasard
    voisin[index] = np.random.choice(medecins_disponibles)
    
    return voisin

def recherche_tabou(num_iters, num_voisins, max_stagnation, len_tabou, gplan: GestionnairePlanning, sol=None, max_dist=None, planning_initial=None, jours_gras=None, eq=None):
    """
    Recherche un planning qui minimise le critère défini dans definition.py par méthode tabou.
    L'algorithme arrête sa recherche lorsqu'il "stagne": aucune amélioration sur max_stagnation étapes successives.
    Renvoie le meilleur individu trouvé pendant toute la recherche, et la liste des critères obtenus au fil de la recherche.
    """

    tabou = deque(maxlen=len_tabou)

    if sol is None:
        sol = gplan.solution_initiale()
    sol_critere = gplan.calcule_critere(sol)

    meilleur_sol = sol.copy()
    meilleur_critere = sol_critere
    stagnation = 0
    scores = []

    pbar = tqdm.tqdm(range(num_iters))

    for i in range(num_iters):
        voisin_critere_min = float('inf')
        meilleur_voisin = None

        tentatives = 0
        max_tentatives = TENTATIVE_MULT_T*num_voisins
        compteur = 0

        while tentatives < max_tentatives and compteur < num_voisins:
            voisin = planning_voisin(sol, gplan, jours_gras)
            voisin = gplan.forcer_contrainte(voisin)
            tentatives += 1

            if planning_initial is not None and max_dist is not None:
                # Ne considérer que les positions non-vides du planning initial
                masque = planning_initial != -1
                dist = np.sum(voisin[masque] != planning_initial[masque])
                if dist > max_dist:
                    continue
            
            compteur += 1

            voisin_critere = gplan.calcule_critere(voisin)

            # critère d'aspiration : A(f(s)) prend la valeur de la meilleure solution s*
            if voisin.tobytes() not in tabou or voisin_critere < meilleur_critere:
                if voisin_critere < voisin_critere_min:
                    voisin_critere_min = voisin_critere
                    meilleur_voisin = voisin.copy()
        
        # Si on n'a trouvé aucun voisin valide
        if meilleur_voisin is None:
            stagnation += 1
            scores.append(sol_critere)
            continue

        tabou.append(sol.tobytes())
        sol = meilleur_voisin
        sol_critere = voisin_critere_min

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
