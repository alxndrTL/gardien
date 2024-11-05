import random
import numpy as np

from config import *

# TODO faire des check si mdc_dispo est bien non vide avant de sample, et quid du cas ou vide ??
# TODO modifier comments bizarres
# TODO commentaires complets

"""
Permet de manipuler des plannings facilement : création, détection de contrainte,
fixer les contraintes, calculer le critère.

On représente un planning de {gardes, astreintes} par un vecteur de longueur 2D
où D est le nombre total de couples (garde, astreinte) à distribuer.
Chaque nombre qui compose le planning indique quel mdc effectue la garde/astreinte concernée.
On peut facilement en déduire le planning des gardes et celui des astreintes en coupant en 2.

La contrainte pour un planning donné est d'empêcher un mdc de travailler au lendemain d'une garde.
Il ne peut pas donc enchaîne une autre garde, ou bien même une autre astreinte.
On implémente cela avec la fonction detecte_contrainte et forcer_contrainte.

Le critère pour un planning donné évalue à quel point le planning respecte les préférences des mdc.
preferences est un tableau de taille (N, D) avec N le nombre de mdc, D le nombre de gardes.
Chaque mdc indique pour chaque garde sa préférence de travail : négatif = ne veut pas, positif = veut.
On implémente cela avec la foncton calcule_critere.
"""
class GestionnairePlanning:
    def __init__(self, nombre_mdc, nombre_gardes, preferences=None, reductions=None, attributs=None, jours_gras=None, planning_initial=None):
        self.N = nombre_mdc
        self.D = nombre_gardes # aussi égal à nombre_astreintes

        self.jours_gras = jours_gras if jours_gras else {'garde': [], 'astreinte': []}
        self.planning_initial = planning_initial.copy() if planning_initial else None

        # preferences est un tableau de taille (NxD).
        # ie chaque mdc spécifie sa préférence concernant chaque jour
        if preferences is None:
            while True:
                preferences = np.random.randint(low=-1, high=3, size=(self.N, self.D))
                if not np.any(np.all(preferences == -1, axis=0)):
                    break
        self.preferences = preferences

        if reductions is None:
            reductions = np.ones((self.N,))
        self.reductions = reductions

        if attributs is None:
            attributs = {}  # dictionnaire vide si pas d'attributs
        self.attributs = attributs  # format: {nom_attribut: list[bool]} où list[bool] de taille N

    def random_mdc(self):
        """
        Renvoie un médecin au hasard (sous forme d'entier entre 0 et N-1).
        """

        return random.randint(0, self.N-1)

    def detecte_contrainte(self, planning):
        """
        Dans un planning, retourne True si :
        -la contrainte de "jour off après la garde" n'est pas respectée
        (ie, un mdc ne doit pas travailler (ni garde ni astreinte) après une garde)
        -un médecin est réaffecté à un jour en gras où il était déjà affecté dans le planning initial
        """

        planning_gardes = planning[:self.D]
        planning_astreintes = planning[self.D:]

        if planning_gardes[0] == planning_astreintes[0]:
            return True

        # Vérifier qu'aucune modification n'est faite avant le jour de début autorisé
        if self.planning_initial is not None and self.jours_gras.get('jour_debut_modif', -1) != -1:
            jour_debut = self.jours_gras['jour_debut_modif']
            for t in range(jour_debut):
                if planning_gardes[t] != self.planning_initial[t] or planning_astreintes[t] != self.planning_initial[self.D + t]:
                    return True
        
        if ENABLE_OFF_AFTER_GARDE:
            for (last_mdc, mdc, mdc_astreinte) in zip(planning_gardes, planning_gardes[1:], planning_astreintes[1:]):
                if last_mdc == mdc:
                    return True
                elif last_mdc == mdc_astreinte:
                    return True
        
        for (mdc_garde, mdc_astreinte) in zip(planning_gardes, planning_astreintes):
            if mdc_garde == mdc_astreinte:
                return True
        
        # Contrainte de non réaffectation des médecins aux jours en gras
        if self.planning_initial is not None:
            planning_initial_gardes = self.planning_initial[:self.D]
            planning_initial_astreintes = self.planning_initial[self.D:]

            for jour in self.jours_gras['garde']:
                if planning_gardes[jour] == planning_initial_gardes[jour]:
                    return True
            
            for jour in self.jours_gras['astreinte']:
                if planning_astreintes[jour] == planning_initial_astreintes[jour]:
                    return True
        
        return False

    def forcer_contrainte(self, planning):
        """
        Dans un planning, détecte si la contrainte de "jour off après la garde" n'est pas respectée
        A chaque fois qu'il y a 2 apparitions consécutives d'un même praticien (GG ou GA), la fonction remplace la 2nd apparition par un autre praticien.
        La planning renvoyé respecte la contrainte.

        force le respect des contraintes:
        - "jour off après la garde"
        - non réaffectation des médecins aux jours en gras
        """

        planning_gardes = planning[:self.D].copy()
        planning_astreintes = planning[self.D:].copy()

        for t in range(len(planning_gardes)):
            if self.planning_initial is not None and t in self.jours_gras['garde']:
                if planning_gardes[t] == self.planning_initial[t]:
                    mdc_dispo = list(range(self.N)) # liste des mdc parmis lesquels on va tirer au sort
                    mdc_dispo.remove(planning_gardes[t]) 
                    planning_gardes[t] = random.choice(mdc_dispo)

            if self.planning_initial is not None and t in self.jours_gras['astreinte']:
                if planning_astreintes[t] == self.planning_initial[self.D + t]:
                    mdc_dispo = list(range(self.N)) # liste des mdc parmis lesquels on va tirer au sort
                    mdc_dispo.remove(planning_astreintes[t]) 
                    planning_astreintes[t] = random.choice(mdc_dispo)

        if ENABLE_OFF_AFTER_GARDE:
            for t in range(1, len(planning_gardes)):
                # un mdc fait deux gardes d'affilé (GG)
                if planning_gardes[t] == planning_gardes[t-1]:
                    mdc_dispo = list(range(self.N)) # liste des mdc parmis lesquels on va tirer au sort
                    mdc_dispo.remove(planning_gardes[t]) # on retire celui qui est en jour off
                    
                    if t < len(planning_gardes) - 1: # il faut faire attente à ne pas engager un mdc qui a une garde le lendemain
                        if planning_gardes[t+1] in mdc_dispo:
                            mdc_dispo.remove(planning_gardes[t+1])

                    # Si c'est un jour en gras, interdire de réutiliser le médecin initial
                    if self.planning_initial is not None and t in self.jours_gras['garde']:
                        mdc_initial = self.planning_initial[t]
                        if mdc_initial != -1 and mdc_initial in mdc_dispo:
                            mdc_dispo.remove(mdc_initial)
                    
                    planning_gardes[t] = random.choice(mdc_dispo) # on tire au sort un mdc

                # un mdc fait une garde suivie par une astreinte (GA)
                if planning_astreintes[t] == planning_gardes[t-1]:
                    mdc_dispo = list(range(self.N)) # liste des mdc parmis lesquels on va tirer au sort
                    mdc_dispo.remove(planning_astreintes[t]) # on retire celui qui est en jour off
                    
                    if t < len(planning_gardes) - 1: # il faut faire attente à ne pas engager un mdc qui a une garde le lendemain
                        if planning_gardes[t+1] in mdc_dispo:
                            mdc_dispo.remove(planning_gardes[t+1])

                    # Si c'est un jour en gras, interdire de réutiliser le médecin initial
                    if self.planning_initial is not None and t in self.jours_gras['astreinte']:
                        mdc_initial = self.planning_initial[self.D + t]
                        if mdc_initial != -1 and mdc_initial in mdc_dispo:
                            mdc_dispo.remove(mdc_initial)
                    
                    planning_astreintes[t] = random.choice(mdc_dispo) # on tire au sort un mdc

        for t in range(len(planning_gardes)):
            if planning_gardes[t] == planning_astreintes[t]:
                mdc_dispo = list(range(self.N))
                mdc_dispo.remove(planning_gardes[t])  # Retire le médecin qui fait déjà la garde

                if t > 0 and planning_gardes[t-1] in mdc_dispo:
                    mdc_dispo.remove(planning_gardes[t-1])

                # Si c'est un jour en gras, interdire de réutiliser le médecin initial
                if self.planning_initial is not None and t in self.jours_gras['astreinte']:
                    mdc_initial = self.planning_initial[self.D + t]
                    if mdc_initial != -1 and mdc_initial in mdc_dispo:
                        mdc_dispo.remove(mdc_initial)

                # Assure qu'il y a un médecin disponible
                if mdc_dispo:
                    planning_astreintes[t] = random.choice(mdc_dispo)
                else:
                    raise Exception(f"Aucun médecin disponible pour l'astreinte au temps {t}")
                
        # Forcer les jours avant jour_debut_modif à rester identiques au planning initial
        if self.planning_initial is not None and self.jours_gras.get('jour_debut_modif', -1) != -1:
            jour_debut = self.jours_gras['jour_debut_modif']
            planning_gardes[:jour_debut] = self.planning_initial[:jour_debut]
            planning_astreintes[:jour_debut] = self.planning_initial[self.D:self.D + jour_debut]

        return np.concatenate((planning_gardes, planning_astreintes))

    def solution_initiale(self):
        """
        Renvoie un planning généré aléatoirement (mais qui respecte la contrainte jour off)
        """
        A = [self.random_mdc() for _ in range(2*self.D)] # construction du planning : d'abord les gardes puis les astreintes
        B = self.forcer_contrainte(A) # on applique la contrainte
        return np.array(B)

    def calcule_critere(self, planning):
        """
        Renvoie la valeur du critère pour le planning donné.
        """

        planning_gardes = planning[:self.D]
        planning_astreintes = planning[self.D:]

        critere = 0

        # respect des préférences des mdc qui ont une garde
        preferences_mdc_gardes = self.preferences[planning_gardes, np.arange(self.D)]
        for pref in preferences_mdc_gardes:
            if pref < 0:
                critere += PENALITE_CRITERE_PREF_NEG*(pref**2)
            elif pref == 0:
                critere += PENALITE_CRITERE_PREF_NULLE
            else:
                critere -= BONUS_CRITERE_PREF_POS*pref**2

        # respect des préférences des mdc qui ont une astreinte (seulement si grosse préf négative)
        preferences_mdc_astreintes = self.preferences[planning_astreintes, np.arange(self.D)]
        for pref in preferences_mdc_astreintes:
            if pref < -5:
                critere += PENALITE_CRITERE_PREF_NEG*(pref**2)

        # Pénalités pour les attributs non respectés
        for _, (mdc_garde, mdc_astreinte) in enumerate(zip(planning_gardes, planning_astreintes)):
            critere += self.penalite_attributs(mdc_garde, mdc_astreinte)

        # autres contraintes ("soft")
        critere += self.calcule_soft_critere(planning)

        return critere

    def calcule_soft_critere(self, planning):
        """
        Renvoie la valeur du critère pour le planning donné.
        """

        planning_gardes = planning[:self.D]
        planning_astreintes = planning[self.D:]

        critere = 0

        # pénaliser les trop petits écarts entre chaque garde
        for mdc in range(self.N):
            jours_mdc = np.where(planning_gardes == mdc)[0]
            if len(jours_mdc) > 1:
                ecarts = np.diff(jours_mdc)
                for ecart in ecarts:
                    if ecart < PETIT_ECART:
                        critere += PENALITE_CRITERE_PETIT_ECART
                    critere += PENALITE_CRITERE_ECART / ecart

        # pénaliser une mauvaise répartition des gardes/astreintes entre les mdc
        nb_positifs_par_mdc = (self.preferences > 0).sum(axis=1) # (N,)
        # gardes
        target_nb_gardes_par_mdc = self.D * ((nb_positifs_par_mdc/self.reductions) / np.sum(nb_positifs_par_mdc/self.reductions)) # (N,)    

        nb_gardes_par_mdc = np.zeros(self.N, dtype=int)
        for mdc in planning_gardes:
            nb_gardes_par_mdc[mdc] += 1

        critere += PENALITE_CRITERE_MAUVAISE_REPART * np.sum((target_nb_gardes_par_mdc - nb_gardes_par_mdc)**2)

        # idem pour les astreintes
        target_nb_astreintes_par_mdc = self.D * ((nb_positifs_par_mdc/self.reductions) / np.sum(nb_positifs_par_mdc/self.reductions)) # (N,)

        nb_astreintes_par_mdc = np.zeros(self.N, dtype=int)
        for mdc in planning_astreintes:
            nb_astreintes_par_mdc[mdc] += 1

        critere += PENALITE_CRITERE_MAUVAISE_REPART * np.sum((target_nb_astreintes_par_mdc - nb_astreintes_par_mdc)**2)

        return critere
    
    def penalite_attributs(self, mdc_garde, mdc_astreinte):
        """
        Calcule la pénalité pour un jour donné en fonction des attributs des médecins de garde et d'astreinte.
        Retourne une pénalité élevée si certains attributs ne sont pas couverts par au moins un des deux médecins.
        """
        penalite = 0
        
        # Pour chaque attribut requis
        for _, mdc_avec_attribut in self.attributs.items():
            # Au moins un des deux médecins doit avoir l'attribut
            if not (mdc_avec_attribut[mdc_garde] or mdc_avec_attribut[mdc_astreinte]):
                penalite += PENALITE_CRITERE_ATTRIBUT_MANQUANT # même ordre de grandeur que les préférences très négatives
        
        return penalite
    
    def distance_sol(self, planning, planning_ref):
        planning_gardes = planning[:self.D]
        planning_astreintes = planning[self.D:]

        planning_ref_gardes = planning_ref[:self.D]
        planning_ref_astreintes = planning_ref[self.D:]

        distance = 0
        distance += np.sum(planning_gardes != planning_ref_gardes)
        distance += np.sum(planning_astreintes != planning_ref_astreintes)

        return distance
    
    def infos_planning(self, planning):
        """
        Calcule différentes caractéristiques d'un planning
        calculer le nombre de -1 attribués en garde (idéalement 0)
        calculer le nombre de 0 attribués en garde (idéalement 0)
        calculer le nb de gardes par mdc
        calculer le nb d'astreintes par mdc
        """

        resultat = {}
        
        # ------ nombre de neg, -1 et 0 attribués ------
        planning_gardes = planning[:self.D]
        planning_astreintes = planning[self.D:]

        nombre_neg = 0
        nombre_moins1 = 0
        nombre_0 = 0

        preferences_mdc_gardes = self.preferences[planning_gardes, np.arange(self.D)]
        for i, pref in enumerate(preferences_mdc_gardes):
            if pref < 0:
                nombre_neg += 1
                if pref == -1:
                    nombre_moins1 += 1
            elif pref == 0:
                nombre_0 += 1

        resultat.update({"nombre_neg": nombre_neg, "nombre_moins1": nombre_moins1, "nombre_0": nombre_0})

        # ------ nombre de gardes et astreintes attribuées par mdc ------

        nb_gardes_par_mdc = np.zeros(self.N, dtype=int)
        nb_astreintes_par_mdc = np.zeros(self.N, dtype=int)

        for mdc in planning_gardes:
            nb_gardes_par_mdc[mdc] += 1

        for mdc in planning_astreintes:
            nb_astreintes_par_mdc[mdc] += 1

        resultat["nb_gardes_par_mdc"] = nb_gardes_par_mdc.tolist()
        resultat["nb_astreintes_par_mdc"] = nb_astreintes_par_mdc.tolist()

        # ------ écart moyen entre les gardes pour chaque mdc ------

        ecart_moyen_gardes_par_mdc = []

        for mdc in range(self.N):
            # Trouver les jours où le mdc a une garde
            jours_mdc = np.where(np.array(planning_gardes) == mdc)[0]
            if len(jours_mdc) > 1:
                # Calculer les écarts entre les gardes successives
                ecarts = np.diff(jours_mdc)
                # Calculer l'écart moyen
                ecart_moyen = np.mean(ecarts)
            else:
                # S'il y a une seule garde ou aucune, l'écart moyen est défini comme None
                ecart_moyen = None
            ecart_moyen_gardes_par_mdc.append(ecart_moyen)

        resultat["ecart_moyen_gardes_par_mdc"] = ecart_moyen_gardes_par_mdc

        return resultat
