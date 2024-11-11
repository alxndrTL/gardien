import random
import numpy as np

"""
[NON UTILISE PAR GARDIEN]

Plusieurs choix ont été faits:

- codage du génotype: même codage que le planning, ie une liste qui accole les gardes et les astreintes.
Pour chaque garde, un entier qui représente le médecin qui effectue la garde, idem pour les astreintes.
Par exemple, [3, 4, 5, 6, 1, 3] signifie qu'il y a 3 paires (garde, astreinte) à attribuer,
et que le mdc 3 fait la première garde tandis que 6 fait la première astreinte.
4 seconde garde, 1 seconde astreinte, etc.

- fitness: cf fonction calcule_critere dans definitions.py. Cette fitness doit être minimisée.

- sélection: on choisit la sélection par tournoi (on tire au hasard k individus et on sélectionne le meilleur)

- élitisme: on conserve quoi qu'il arrive le meilleur individu pour la génération suivante
(cela permet de ne pas perdre une potentielle bonne solution)

- croisements: on a implémenté les one-point-crossover et two-point-crossover (pour l'instant)

- mutations : on a implémenté la mutation par substitution et par échange (pour l'instant)

sources (sélection par tournoi):
https://perso.liris.cnrs.fr/alain.mille/enseignements/master_ia/rapports_2006/Programmation%20Genetique_4p.pdf
"""

# croisements

def one_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    enfant1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    enfant2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return enfant1, enfant2

def two_point_crossover(parent1, parent2):
    crossover_points = sorted(np.random.choice(range(1, 2*len(parent1)), size=2, replace=False))
    
    mask = np.ones(len(parent1), dtype=bool)
    mask[crossover_points[0]:crossover_points[1]] = False
    
    enfant1 = np.where(mask, parent1, parent2)
    enfant2 = np.where(mask, parent2, parent1)
    return enfant1, enfant2

# mutations

def mutation_substitution(individu, gplan):
    # indice + valeur
    indice_a_muter = np.random.randint(0, len(individu))
    nouvelle_valeur = gplan.random_mdc()

    # construction de l'individu muté
    individu_mute = individu.copy()
    individu_mute[indice_a_muter] = nouvelle_valeur
    return individu_mute

def mutation_echange(individu):
    # les deux indices à échanger
    indices = np.random.choice(len(individu), size=2, replace=False)

    individu_mute = individu.copy()
    individu_mute[indices[0]], individu_mute[indices[1]] = individu_mute[indices[1]], individu_mute[indices[0]]
    return individu_mute

def recherche_algo_genetique(taille_population, nb_generations, taux_mutation, gplan, verbose=False):
    """
    Recherche un planning qui minimise le critère défini dans definition.py par algorithme génétique.
    Renvoie le meilleur individu trouvé pendant toute la recherche, et la liste des critères obtenus au fil de la recherche.
    """

    # meilleur fitness trackée à chaque génération
    scores = []

    # init population
    population = [gplan.forcer_contrainte(gplan.solution_initiale()) for _ in range(taille_population)]
    #population = [gplan.forcer_contrainte(gplan.solution_manuel()) for _ in range(taille_population)]

    meilleur_score = min(gplan.calcule_critere(ind) for ind in population)
    scores.append(meilleur_score)

    if verbose:
        print(f"Génération 0: Meilleur score = {meilleur_score}")

    for generation in range(nb_generations):
        fitness = [gplan.calcule_critere(individu) for individu in population]
        
        # sélection des parents par tournoi
        def selection_tournoi(k=3):
            indices_tournoi = random.sample(range(taille_population), k)
            return population[min(indices_tournoi, key=lambda i: fitness[i])]

        # sélection des parents par roulette (après plusieurs tests, les tournois sont meilleurs)
        def selection_roulette():
            somme_fitness = sum(fitness)
            
            rand = random.uniform(0, somme_fitness)
            
            somme_courante = 0
            for i, f in enumerate(fitness):
                somme_courante += f
                if somme_courante > rand:
                    return population[i]
            
            # (au cas ou)
            return population[-1]
        
        nouvelle_population = []
        
        # élitisme : on conserve le meilleur individu
        meilleur_index = fitness.index(min(fitness))
        nouvelle_population.append(population[meilleur_index])
        
        # nouvelle génération
        while len(nouvelle_population) < taille_population:
            parent1 = selection_tournoi()
            parent2 = selection_tournoi()
            #parent1 = selection_roulette() # la selection par roulette ne donne pas de bons resultats
            #parent2 = selection_roulette()
            
            # croisements
            if random.random() < 0.5:
                enfant1, enfant2 = one_point_crossover(parent1, parent2)
            else:
                enfant1, enfant2 = two_point_crossover(parent1, parent2)
            
            # mutations
            if random.random() < taux_mutation:
                enfant1 = mutation_substitution(enfant1, gplan)
            if random.random() < taux_mutation:
                enfant2 = mutation_echange(enfant2)
            
            # les enfants doivent être "viables"
            enfant1 = gplan.forcer_contrainte(enfant1)
            enfant2 = gplan.forcer_contrainte(enfant2)
            
            nouvelle_population.extend([enfant1, enfant2])
    
        population = nouvelle_population[:taille_population]
    
        meilleur_score = min(gplan.calcule_critere(ind) for ind in population)
        scores.append(meilleur_score)

        if verbose:
            print(f"Génération {generation + 1}: Meilleur score = {meilleur_score}")

    meilleur_individu = min(population, key=lambda ind: gplan.calcule_critere(ind))

    return meilleur_individu, scores
