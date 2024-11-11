import random
import numpy as np

from config import *

"""
Permet de manipuler des plannings facilement : création, détection de contrainte, fixer les contraintes, calculer le critère.

On représente un planning de {gardes, astreintes} par un vecteur de longueur 2D
où D est le nombre total de couples (garde, astreinte) à distribuer.
Chaque nombre qui compose le planning indique quel mdc effectue la garde/astreinte concernée.
On peut facilement en déduire le planning des gardes et celui des astreintes en coupant en 2.

Les contraintes dures pour un planning donné sont:
- d'empêcher un mdc de travailler au lendemain d'une garde (désactivable dans config.py)
- d'empêcher un médecin de faire garde+astreinte en même temps
- ne pas modifier un médecin d'une garde ou astreinte "soulignée", c'est-à-dire fixée par l'utilisateur
- modifier un médecin d'une garde ou astreinte "en gras", c'est-à-dire à modifier (selon l'utilisateur)
On implémente cela avec les fonctions detecte_contrainte et forcer_contrainte.

Le critère pour un planning donné est :
- à quel point le planning respecte les préférences des mdc
- les attributs sont correctement représentés
- un "soft critère" (cf calcule_soft_critere) qui calcule des choses plus subtiles comme la distribution des gardes, les écarts etc

Les préférences sont représentées par un tableau de taille (N, D) avec N le nombre de mdc, D le nombre de gardes.
Chaque mdc indique pour chaque garde sa préférence de travail : négatif = ne veut pas, positif = veut. (avec graduation possible)
Note : les préférences concernant les gardes et non les astreintes. Toutefois, une préférence très fortement négative évitera aussi les astreintes. (cf SEUIL_PREF_NEG_ASTREINTE)
On implémente cela avec la foncton calcule_critere.
"""
class GestionnairePlanning:
    def __init__(self, nombre_mdc, nombre_gardes, preferences=None, reductions=None, attributs=None, implications=None, jours_gras=None, jours_soulignes=None, planning_initial=None):
        self.N = nombre_mdc
        self.D = nombre_gardes # aussi égal au nombre d'astreintes

        self.jours_gras = jours_gras if jours_gras else {'garde': [], 'astreinte': []} # jours à modifier
        self.jours_soulignes = jours_soulignes if jours_soulignes else {'garde': [], 'astreinte': []} # jours à fixer
        self.planning_initial = planning_initial.copy() if planning_initial else None # on garde le planning initial en mémoire pour calculer la distance

        # preferences est un tableau de taille (NxD).
        # ie chaque mdc spécifie sa préférence concernant chaque jour
        self.preferences = preferences
        
        # reductions compte, pour chaque mdc, le nombre d'équipes auquel il appartient
        # permet alors de réduire son implication pour chaque planning
        # [NOTE : comportement désactivé lorsque des nombres cibles de gardes et d'astreintes sont donnés]
        if reductions is None:
            reductions = np.ones((self.N,))
        self.reductions = reductions

        # attributs des médecins (ex: CCV)
        if attributs is None:
            attributs = {}  # dictionnaire vide si pas d'attributs
        self.attributs = attributs  # format: {nom_attribut: list[bool]} où list[bool] de taille N

        # répertorie les implications pour chaque mdc (nombre cible de gardes et nombre cible d'astreintes à distribuer)
        self.implications = implications

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
        -un médecin travaille en garde et en astreinte le même jour
        -un médecin est réaffecté à un jour en gras où il était déjà affecté dans le planning initial
        -un médecin a été déplacé d'une garde/astreinte soulignée (fixée)

        Si aucune contrainte n'est pas respectée, return False
        """

        planning_gardes = planning[:self.D]
        planning_astreintes = planning[self.D:]

        # si le premier jour, le même mdc fait garde et astreinte -> True
        if planning_gardes[0] == planning_astreintes[0]:
            return True
        
        # vérification contrainte "jour off"
        if ENABLE_OFF_AFTER_GARDE:
            for (last_mdc, mdc, mdc_astreinte) in zip(planning_gardes, planning_gardes[1:], planning_astreintes[1:]):
                if last_mdc == mdc:
                    return True
                elif last_mdc == mdc_astreinte:
                    return True
                
        # vérification contrainte "médecin fait garde et astreinte"
        for (mdc_garde, mdc_astreinte) in zip(planning_gardes, planning_astreintes):
            if mdc_garde == mdc_astreinte:
                return True
        
        # vérification contrainte de non réaffectation des jours gras
        if self.planning_initial is not None:
            planning_initial_gardes = self.planning_initial[:self.D]
            planning_initial_astreintes = self.planning_initial[self.D:]

            for jour in self.jours_gras['garde']:
                if planning_gardes[jour] == planning_initial_gardes[jour]:
                    return True
            
            for jour in self.jours_gras['astreinte']:
                if planning_astreintes[jour] == planning_initial_astreintes[jour]:
                    return True
                
        # vérification contrainte de fixation des jours soulignés
        if self.planning_initial is not None:
            planning_initial_gardes = self.planning_initial[:self.D]
            planning_initial_astreintes = self.planning_initial[self.D:]

            for jour in self.jours_soulignes['garde']:
                if planning_gardes[jour] != planning_initial_gardes[jour]:
                    return True
                
            for jour in self.jours_soulignes['astreinte']:
                if planning_astreintes[jour] != planning_initial_astreintes[jour]:
                    return True
        
        return False

    def forcer_contrainte(self, planning):
        """
        Force le respect des contraintes:
        - jour off après la garde
        - médecin différent pour la garde et pour l'astreinte d'un même jour
        - non réaffectation des médecins aux jours en gras
        - non modifications des médecins aux jours soulignés

        Le forçage de contrainte se fait en remplaçant les médecins des jours problématiques par d'autres médecins (qui conviennent)
        La planning renvoyé respecte la contrainte.
        """

        planning_gardes = planning[:self.D].copy()
        planning_astreintes = planning[self.D:].copy()

        # empeche la réaffection des médecins aux jours en gras
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
        
        # forcer les jours soulignés à rester identiques au planning initial
        for t in range(len(planning_gardes)):
            if self.planning_initial is not None and t in self.jours_soulignes['garde']:
                if planning_gardes[t] != self.planning_initial[t]:
                    planning_gardes[t] = self.planning_initial[t]

            if self.planning_initial is not None and t in self.jours_soulignes['astreinte']:
                if planning_astreintes[t] != self.planning_initial[self.D + t]:
                    planning_astreintes[t] = self.planning_initial[self.D + t]

        # forcer la contrainte "jour off après la garde"
        if ENABLE_OFF_AFTER_GARDE:
            for t in range(1, len(planning_gardes)):

                # un mdc fait deux gardes d'affilé (GG)
                if planning_gardes[t] == planning_gardes[t-1] and t in self.jours_soulignes['garde']:
                    if t-1 in self.jours_soulignes['garde']:
                        break
                    else:
                        mdc_dispo = list(range(self.N))
                        mdc_dispo.remove(planning_gardes[t])
                        planning_gardes[t-1] = random.choice(mdc_dispo)
                if planning_gardes[t] == planning_gardes[t-1] and t not in self.jours_soulignes['garde']:
                    mdc_dispo = list(range(self.N)) # liste des mdc parmis lesquels on va tirer au sort
                    mdc_dispo.remove(planning_gardes[t]) # on retire celui qui est en jour off
                    
                    if t < len(planning_gardes) - 1: # il faut faire attente à ne pas engager un mdc qui a une garde le lendemain
                        if planning_gardes[t+1] in mdc_dispo:
                            mdc_dispo.remove(planning_gardes[t+1])

                    # si c'est un jour en gras, interdire de réutiliser le médecin initial
                    if self.planning_initial is not None and t in self.jours_gras['garde']:
                        mdc_initial = self.planning_initial[t]
                        if mdc_initial != -1 and mdc_initial in mdc_dispo:
                            mdc_dispo.remove(mdc_initial)

                    # si le lendemain est un jour souligné, gérer le cas
                    if self.planning_initial is not None and (t < len(planning_gardes)-1):
                        if t+1 in self.jours_soulignes['garde']:
                            if self.planning_initial[t+1] in mdc_dispo:
                                mdc_dispo.remove(self.planning_initial[t+1]) # on empeche de prendre un mdc qui a une garde fixé le lendemain 
                        if t+1 in self.jours_soulignes['astreinte']:
                            if self.planning_initial[self.D+t+1] in mdc_dispo:
                                mdc_dispo.remove(self.planning_initial[self.D+t+1]) # on empeche de prendre un mdc qui a une astreinte fixée le lendemain
                    
                    if mdc_dispo:
                        planning_gardes[t] = random.choice(mdc_dispo) # on tire au sort un mdc
                    else:
                        raise Exception(f"\033[1m\033[31m[ERREUR]\033[0m Aucun médecin disponible pour l'astreinte au temps {t}. Vous pouvez essayer de relancer l'algorithme.")


                # un mdc fait une garde suivie par une astreinte (GA)
                if planning_astreintes[t] == planning_gardes[t-1] and t in self.jours_soulignes['astreinte']:
                    if t-1 in self.jours_soulignes['garde']:
                        break
                    else:
                        mdc_dispo = list(range(self.N))
                        mdc_dispo.remove(planning_astreintes[t])
                        planning_gardes[t-1] = random.choice(mdc_dispo)
                if planning_astreintes[t] == planning_gardes[t-1] and t not in self.jours_soulignes['astreinte']:
                    mdc_dispo = list(range(self.N)) # liste des mdc parmis lesquels on va tirer au sort
                    mdc_dispo.remove(planning_astreintes[t]) # on retire celui qui est en jour off
                    
                    if t < len(planning_gardes) - 1: # il faut faire attente à ne pas engager un mdc qui a une garde le lendemain
                        if planning_gardes[t+1] in mdc_dispo:
                            mdc_dispo.remove(planning_gardes[t+1])

                    # si c'est un jour en gras, interdire de réutiliser le médecin initial
                    if self.planning_initial is not None and t in self.jours_gras['astreinte']:
                        mdc_initial = self.planning_initial[self.D + t]
                        if mdc_initial != -1 and mdc_initial in mdc_dispo:
                            mdc_dispo.remove(mdc_initial)
                    
                    if mdc_dispo:
                        planning_astreintes[t] = random.choice(mdc_dispo) # on tire au sort un mdc
                    else:
                        raise Exception(f"\033[1m\033[31m[ERREUR]\033[0m Aucun médecin disponible pour l'astreinte au temps {t}. Vous pouvez essayer de relancer l'algorithme.")

        # forcer la contrainte "même médecin en garde et en astreinte"
        for t in range(len(planning_gardes)):
            if planning_gardes[t] == planning_astreintes[t]:
                mdc_dispo = list(range(self.N))
                mdc_dispo.remove(planning_gardes[t])  # Retire le médecin qui fait déjà la garde

                if t > 0 and planning_gardes[t-1] in mdc_dispo:
                    mdc_dispo.remove(planning_gardes[t-1])

                # si c'est un jour en gras, interdire de réutiliser le médecin initial
                if self.planning_initial is not None and t in self.jours_gras['astreinte']:
                    mdc_initial = self.planning_initial[self.D + t]
                    if mdc_initial != -1 and mdc_initial in mdc_dispo:
                        mdc_dispo.remove(mdc_initial)

                if mdc_dispo:
                    planning_astreintes[t] = random.choice(mdc_dispo)
                else:
                    raise Exception(f"\033[1m\033[31m[ERREUR]\033[0m Aucun médecin disponible pour l'astreinte au temps {t}. Vous pouvez essayer de relancer l'algorithme.")

        return np.concatenate((planning_gardes, planning_astreintes))

    def solution_initiale(self):
        """
        Renvoie un planning généré aléatoirement (mais qui respecte les contraintes)
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
        for jour in range(self.D):
            if jour not in self.jours_soulignes['garde']:
                mdc = planning_gardes[jour]
                pref = self.preferences[mdc, jour]
                if pref < 0:
                    critere += PENALITE_CRITERE_PREF_NEG*(pref**2)
                elif pref == 0:
                    critere += PENALITE_CRITERE_PREF_NULLE
                else:
                    critere -= BONUS_CRITERE_PREF_POS*pref**2

        # respect des préférences des mdc qui ont une astreinte (seulement si grosse préf négative, <SEUIL_PREF_NEG_ASTREINTE)
        for jour in range(self.D):
            if jour not in self.jours_soulignes['astreinte']:
                mdc = planning_astreintes[jour]
                pref = self.preferences[mdc, jour]
                if pref < SEUIL_PREF_NEG_ASTREINTE:
                    critere += PENALITE_CRITERE_PREF_NEG*(pref**2)

        # pénalités pour les attributs non respectés
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
        # (cf guide d'utilisateur pour plus de détais sur cette stratégie)
        if self.implications is None:
            nb_positifs_par_mdc = (self.preferences > 0).sum(axis=1) # (N,)
            target_nb_gardes_par_mdc = self.D * ((nb_positifs_par_mdc/self.reductions) / np.sum(nb_positifs_par_mdc/self.reductions)) # (N,)
            target_nb_astreintes_par_mdc = self.D * ((nb_positifs_par_mdc/self.reductions) / np.sum(nb_positifs_par_mdc/self.reductions)) # (N,)
        else:
            target_nb_gardes_par_mdc = self.implications['gardes']
            target_nb_astreintes_par_mdc = self.implications['astreintes']   

        nb_gardes_par_mdc = np.zeros(self.N, dtype=int)
        for mdc in planning_gardes:
            nb_gardes_par_mdc[mdc] += 1

        critere += PENALITE_CRITERE_MAUVAISE_REPART * np.sum((target_nb_gardes_par_mdc - nb_gardes_par_mdc)**2)

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
        
        # loop sur les attributs
        for _, mdc_avec_attribut in self.attributs.items():
            # au moins un des deux médecins doit avoir l'attribut
            if not (mdc_avec_attribut[mdc_garde] or mdc_avec_attribut[mdc_astreinte]):
                penalite += PENALITE_CRITERE_ATTRIBUT_MANQUANT # même ordre de grandeur que les préférences très négatives
        
        return penalite
    
    def distance_sol(self, planning, planning_ref):
        """
        Renvoie la distance entre deux plannings.
        (distance de Hamming, cf papier section 6.2.2)
        """

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
        Donne des informations générales sur le planning [NON UTILISE PAR GARDIEN]
        
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
                ecarts = np.diff(jours_mdc) # écarts entre les gardes successives
                ecart_moyen = np.mean(ecarts) # écart moyen
            else:
                ecart_moyen = None
            ecart_moyen_gardes_par_mdc.append(ecart_moyen)

        resultat["ecart_moyen_gardes_par_mdc"] = ecart_moyen_gardes_par_mdc

        return resultat
