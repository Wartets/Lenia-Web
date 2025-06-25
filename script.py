import os # Opérations sur les fichiers et répertoires
import re # Opérations sur les expressions régulières
import time # Fonctions liées au temps
import random as rd # Génération de nombres aléatoires
import numpy as np # Opérations numériques
from numpy import float32, int32
import matplotlib.pyplot as plt # Bibliothèque de traçage
import matplotlib.animation as animation # Création d'animations
import matplotlib.patheffects as path_effects # Effets pour les chemins
import matplotlib.colors as mcolors # Spécifications des couleurs
from mpl_toolkits.axes_grid1 import make_axes_locatable # Localisation des axes d'un graphique
import scipy as sp # Calculs scientifiques
from tqdm import tqdm # Barre de progression
import multiprocessing # Multiprocessing pour la parallélisation
from functools import lru_cache # Optimisation de la mise en cache
from typing import Dict, Any, List, Callable # Support d'exécution pour les indications de type
from moviepy.editor import VideoClip

# Définir les chemins relatifs
base_path = '' # emplacement de base
simulation_path = os.path.join(base_path, 'Simulation')
initialisation_path = os.path.join(base_path, 'Initialisation')
colormap_path = os.path.join(base_path, 'Colormap')

os.makedirs(simulation_path, exist_ok=True)
os.makedirs(initialisation_path, exist_ok=True)
os.makedirs(colormap_path, exist_ok=True)

# Optimisation pour des performances optimales
# Cellule à n'activer qu'une fois par démarrage de Kernel Jupyter
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

np.set_printoptions(precision=4)

os.environ['MKL_DYNAMIC'] = 'FALSE'

multiprocessing.set_start_method('spawn')

"""
Création d'un tableau qui va permettre de numéroter les fichiers et determiner le temps de calcul pour les simulations
"""

for i in range (0, 2):
    if not os.path.exists('tab.npy'):
        data = np.array([None, 1, 1])
        """[valeur, effectif, numéro de file]"""
        np.save(os.path.join(base_path, 'tab.npy'), data)
    else:
        tab = np.load('tab.npy', allow_pickle=True)

num_file = tab[2] # numérotation du fichier
    
# print(f'tab = {tab}')

# Définition des colormaps personnalisées
gaga = mcolors.LinearSegmentedColormap.from_list('gaga', np.loadtxt(os.path.join(colormap_path, 'gaga-colormap.txt')), N=256)
doly = mcolors.LinearSegmentedColormap.from_list('doly', np.loadtxt(os.path.join(colormap_path, 'doly-colormap.txt')), N=256)
dolydark = mcolors.LinearSegmentedColormap.from_list('dolydark', np.loadtxt(os.path.join(colormap_path, 'dolydark-colormap.txt')), N=256)
dolymiddle = mcolors.LinearSegmentedColormap.from_list('dolydmiddle', np.loadtxt(os.path.join(colormap_path, 'dolymiddle-colormap.txt')), N=256)
cabry = mcolors.LinearSegmentedColormap.from_list('cabry', np.loadtxt(os.path.join(colormap_path, 'cabry-colormap.txt')), N=2**10) # Oui,  2^10

# Création de la grille pour le tracé
x = np.linspace(0, 1, 2560)
y = np.linspace(0, 1, 256)
X, Y = np.meshgrid(x, y)

# Affichage des dégradés de couleur pour les trois colormaps
fig, axs = plt.subplots(1, 6, figsize=(12, 6))

for ax, cmap, title in zip(axs, [gaga, doly, dolydark, dolymiddle, cabry, 'nipy_spectral'],
                           ['gaga', 'doly','dolydark', 'dolymiddle', 'cabry', 'nipy_spectral']):
    ax.imshow(X, cmap=cmap)
    ax.axis('off')
    ax.set_title(title)

plt.tight_layout()
plt.show()

# Pour éviter d'avoir des soucis lorsque l'on affiche le nom d'une cmap custom
def get_colormap_name(cmap):
    if isinstance(cmap, str):
        return cmap
    else:
        return cmap.name
		
# Valeurs par défaut
DPI = 88 # Qualité en point par pouce # defaut:100 des valeurs plus basses accélèrent le temps de rendu, conseillé: 82
FPS = 25 # Frames par seconde # recommandé: 25
CMAP = cabry # gaga / 'nipy_spectral' / doly / dolydark / dolymiddle / cabry # Choix du color-map
INTERP = 'bicubic' # Les deux interpolations utilisées fréquement
INTERP2 = 'hanning' # 'nearest'

N = 350 # hauteur du graphique # Taille recommandée 300/400
M = N # largeur du graphique # Carré recommandé

dt = 0.1 # "écart de temps" # recommandé: 0.1
R = 13 # valeur défaut, correspond au nombre de cellule par rayon de Kernel

species_files = [ # nom du fichier et paramètres associées
    "Orbium_unicaudatus2.npy",             # R = 13 ; k = bump4()               ; d = gaus(0.15, 0.017) * 0.1
    "Gyropteron_arcus.npy",                # R = 26 ; k = bump4()               ; d = gaus(0.293, 0.0511) * 0.1
    "Scutium_solidus.npy",                 # R = 26 ; k = bump4()               ; d = gaus(0.29, 0.043) * 0.1
    "Hydrogeminium.npy",                   # R = 18 ; k = quad4(1/2,1,2/3)      ; d = quad4(0.26, 0.036) * 0.1
    "SmallBug.npy",                        # R = 13 ; k = bump4()               ; d = gaus(0.31, 0.048) * 0.1
    "TriheliciumPachus.npy",               # R = 13 ; k = stpz1/4()             ; d = stpz(0.46, 0.119) * 0.1
    "DiscutiumPachus.npy",                 # R = 13 ; k = stpz1/4()             ; d = stpz(0.545, 0.186) * 0.1
    "Compilation.npy",                     # R = 13 ; k = bump4()               ; d = gaus(0.337, 0.057) * 0.1
    "CircogeminiumVentilans.npy",          # R = 18 ; k = quad4(1,1,1)          ; d = quad4(0.29, 0.035) * 0.1
    "BigCircogeminiumVentilans.npy",       # R = 45 ; k = quad4(1,1,1)          ; d = quad4(0.29, 0.035) * 0.1
    "GyrogeminiumSerratus.npy",            # R = 36 ; k = quad4(1/2,1,1/2)      ; d = quad4(0.24, 0.03) * 0.1
    "Gyrorbium.npy",                       # R = 13 ; k = bump4()               ; d = gaus(0.156, 0.0224) * 0.1
    "Synorbium.npy",                       # R = 26 ; k = bump4()               ; d = gaus(0.152, 0.0156) * 0.1
    "Triorbium.npy",                       # R = 26 ; k = bump4()               ; d = gaus(0.114, 0.0115) * 0.1
    "Decascutium.npy",                     # R = 26 ; k = bump4()               ; d = gaus(0.48, 0.108) * 0.1
    "CatenoscutiumBidirectus.npy",         # R = 26 ; k = bump4()               ; d = gaus(0.29, 0.043) * 0.1
    "Vagopteron.npy",                      # R = 52 ; k = bump4()               ; d = gaus(0.218, 0.0351) * 0.1
    "HeptapteryxSerratusLiquefaciens.npy", # R = 20 ; k = quad4(3/4,1,1)        ; d = quad4(0.34, 0.051) * 0.1
    "Hexacaudopteryx.npy",                 # R = 26 ; k = quad4()               ; d = quad4(0.35, 0.048) * 0.1
    "CatenopteryxCyclon.npy",              # R = 26 ; k = bump4()               ; d = gaus(0.34, 0.045) * 0.2
    "CatenopteryxCyclonScutoides.npy",     # R = 26 ; k = bump4()               ; d = gaus(0.38, 0.07) * 0.2
    "CatenoheliciumBispiraeScutoides.npy", # R = 26 ; k = bump4()               ; d = gaus(0.407, 0.0806) * 0.1
    "DecadentiumVolubilis.npy",            # R = 72 ; k = quad4(2/3,1,2/3,1/3)  ; d = gaus(0.15, 0.014) * 0.1
    "AerogeminiumQuietus.npy",             # R = 18 ; k = quad4(1,1,1)          ; d = quad4(0.3, 0.048) * 0.1
    "HydrogeminiumNatans2.npy",            # R = 36 ; k = quad4(1,1,1)          ; d = quad4(0.26, 0.036) * 0.1
    "GliderGun.npy",                       # R = 2  ; k = life()                ; d = stpz(0.35, 0.07) * 1
    "Weekender.npy",                       # R = 2  ; k = life()                ; d = stpz(0.35, 0.07) * 1
    "SpaceFiller.npy",                     # R = 2  ; k = life()                ; d = stpz(0.35, 0.07) * 1
    "Pufferfish.npy",                      # R = 2  ; k = life()                ; d = stpz(0.35, 0.07) * 1
    "R-pentomino.npy",                     # R = 2  ; k = life()                ; d = stpz(0.35, 0.07) * 1
    "Hexastrium.npy",                      # R = 96 ; k = quad4(1,1/12,1)       ; d = quad4(0.2, 0.024) * 0.1
    "Fish.npy",                            # R = 10 ; Multiple couples : filter, growth
    "Aquarium_R.npy",                      # R = 12 ; Multiple channels : Red
    "Aquarium_G.npy",                      # R = 12 ; Multiple channels : Green
    "Aquarium_B.npy",                      # R = 12 ; Multiple channels : Blue
    "DodecadentiumNausia.npy",             # R = 54 ; k = quad4(2/3,1,1/3)      ; d = quad4(0.27, 0.033) * 0.1
    "DodecafoliumVentilans.npy",           # R = 72 ; k = bump4(1/2,7/12,3/4,1) ; d = gaus(0.23, 0.019) *0.1
]

species_names = [
    "Orbium Unicaudatus",
    "Gyropteron Arcus",
    "Scutium Solidus",
    "Hydrogeminium",
    "Small Bug",
    "Trihelicium Pachus",
    "Discutium Pachus",
    "Compile d'espèces",
    "Circogeminium Ventilans",
    "Big C. Ventilans",
    "Gyrogeminium Serratus",
plt.show()

# Définir une fonction gaussienne 2D
def gauss(x, mu, sigma):
    return np.exp(-((x - mu)**2) / (2 * sigma**2))

# Définir une fonction gaussienne 3D
def gauss3D(x, y, mu_x, mu_y, sigma_x, sigma_y):
    return np.exp(-0.5 * (((x - mu_x) / sigma_x)**2 + ((y - mu_y) / sigma_y)**2))

# Définir une vraie fonction gaussienne
def gaussvrai(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# Générer un vecteur de 3 éléments avec des valeurs aléatoires dont la somme est égale à 1
def vect3():
    a = np.random.rand()
    b = np.random.rand() * (1 - a)
    c = 1 - a - b
    return [a, b, c]

# Générer un vecteur de n éléments avec des valeurs aléatoires dont la somme est égale à 1
def vectn(n):
    values = np.random.rand(n)
    values /= values.sum()
    return np.array(values.tolist())

# Nettoyer une chaîne en supprimant les parenthèses avec des nombres, "(unworking)" et les espaces
def clean(string):
    string = re.sub(r'\(\d+\)', '', string)
    string = string.replace("(unworking)", "")
    string = string.replace(" ", "")
    return string

# Dénaturer une chaîne en ajoutant des espaces avant les lettres majuscules et en remplaçant "-" et "_" par des espaces
def unclean(string):
    string = re.sub(r'(?<!^)(?=[A-Z])', ' ', string)
    string = string.replace("-", " ").replace("_", " ")
    return string

# Transforme une grille d'une forme [[A, B], [C, D]] en une grille de la forme [[D, C], [B, A]]
def transform_grid(grid):
    N, M = len(grid), len(grid[0])
    
    # Assumant que N et M sont pairs
    mid_N, mid_M = N // 2, M // 2
    
    # Extraction des parties A, B, C, D
    A = grid[:mid_N, :mid_M]
    B = grid[:mid_N, mid_M:]
    C = grid[mid_N:, :mid_M]
    D = grid[mid_N:, mid_M:]
    
    # Reconstruction de la nouvelle grille [[D, C], [B, A]]
    top_row = np.concatenate((D, C), axis=1)
    bottom_row = np.concatenate((B, A), axis=1)
    new_grid = np.concatenate((top_row, bottom_row), axis=0)
    
    return new_grid

# Fonction de base gaussienne particulière
def growth_lenia(X, m, s):
    return 2 * gauss(X, m, s) - 1

# Fonction de Kernel

# Kernel à trois anneaux avec des poids différents pour chaque anneaux
def trimodal(rings=[1, 1, 1], r=R):
    nb_rings = len(rings)
    y, x = np.ogrid[-r:r, -r:r]
    distance = np.sqrt((x)**2 + (y)**2) / r
    # distance = distance * nb_rings
    B1, B2, B3 = rings
    h = np.abs(distance)
    def c(h):
        alpha = 4
        k = 4 * h * (1 - h)
        return k**alpha
    h_new = np.copy(h)
    mask1 = (h < 1/3)
    mask2 = (h >= 1/3) & (h < 2/3)
    mask3 = (h >= 2/3) & (h < 1)
    
    h_new[mask1] = B1 * c(3 * h[mask1])
    h_new[mask2] = B2 * c(3 * h[mask2] - 1)
    h_new[mask3] = B3 * c(3 * h[mask3] - 2)
    h_new[h >= 1] = 0
    return h_new

# Kernel à multi anneaux avec des poids différents
def Multiring(rings=[1/2, 1, 2/3], r=R): # fonction pour un kernel multiring
    global nb_rings
    nb_rings = len(rings)
    pos_y, pos_x = N // 2, M // 2
    y, x = np.ogrid[-r:r, -r:r]
    distance = np.sqrt(x**2 + y**2) / r
    distance = distance * nb_rings
    K_lenia = np.zeros_like(distance)
    for i in range(nb_rings):
        masque = (distance.astype(int) == i)
        K_lenia += masque * rings[i] * gauss(distance %1, 0.5, 0.15) # %1 pour centrer le gaussien sur la grille
    K_lenia = K_lenia / np.sum(K_lenia)
    return K_lenia

# Multi kernel avec plusieurs anneaux
def MultipleGrowth(rings=[0.5, 1, 0.667], r=R):
    bs = [[1,5/12,2/3],[1/12,1],[1]]
    ms = [0.156,0.193,0.342]
    ss = [0.0118,0.049,0.0891]
    ring_strengths = [0.5, 1, 0.667]
    nb_rings = len(ring_strengths)
    y, x = np.ogrid[-r:r, -r:r]
    fKs = []
    for b in bs:
        distance = np.sqrt(x**2 + y**2) / R * len(b)
        K = np.zeros_like(distance)
        mu = 0.5
        sigma = 0.15
        for i in range(len(b)):
            mask = (distance.astype(int) == i)
            K += mask * b[i] * gauss(distance%1, mu, sigma)
        fK = np.fft.fft2(np.fft.fftshift(K / np.sum(K)))
        fKs.append(fK)
    # return fKs
    K_lenia = np.zeros_like(distance)
    for a in (0, nb_rings-1):
        K_lenia += ring_strengths[a]*np.real(fKs[a])
    return transform_grid(np.clip(K_lenia, 0, 1))

# Choix de Kernel

def Kernel_choice(K, rings=[0.5, 1, 0.67], r=R):
    global R
    # print(K, rings, r)
    y, x = np.ogrid[-r:r, -r:r]
    distance = np.sqrt((x+1)**2 + (y+1)**2) / r
    if K == 1: # Kernel Simple Gaussien David Louapre
        alpha = 4
        MU = 0.5
        SIGMA = 0.15
        K_lenia = gauss(distance, MU, SIGMA)
        K_lenia[distance > 1] = 0
        K_lenia = K_lenia / np.sum(K_lenia)  # Normalisation
        return K_lenia
    
    elif K == 2: # Kernel simple gaussien Bert Chan
        alpha = 4
        epsilon = 1e-10
        distance_clipped = np.clip(distance, epsilon, 1 - epsilon)  # Pour éviter la division par zéro
        K_lenia = np.exp(alpha * (1 - 1 / (4 * distance_clipped * (1 - distance_clipped))))
        K_lenia[distance > 1] = 0
        K_lenia = K_lenia / np.sum(K_lenia)  # Normalisation
        return K_lenia
    
    elif K == 3: # Kernel unimodal par step (unworking)
        q = 1/4
        K_lenia = np.where((q <= distance) & (distance <= 1 - q), 1, 0)
        # K_lenia = K_lenia / np.sum(K_lenia)
        return K_lenia
    
    elif K == 4: # Kernel multiring
        return Multiring(rings, r)
    
    elif K == 5: # Kernet Spot gaussien (unworking)
        y, x = np.ogrid[-r:r+1, -r:r+1]
        K_lenia = np.exp(-((x**2 + y**2) / (2 * r**2)))
        K_lenia[K_lenia < np.finfo(K_lenia.dtype).eps * K_lenia.max()] = 0
        return K_lenia / K_lenia.sum()
    
    elif K == 6: # Kernel Trimodal (unworking)
        return trimodal(rings, r)
    
    elif K == 7: # Game of life
        K_lenia = np.array([[1,1,1],[1,0,1],[1,1,1]])
        return K_lenia
    
    elif K == 8: # Multiple couples (filter,growth)
        K_lenia = (MultipleGrowth(rings=[1/2, 1, 2/3], r=r))
        return K_lenia
    
    else:
        raise ValueError("Invalid kernel type")

# Fonction pour faire évoluer le motif lenia en utilisant un noyau simple
def evolve_lenia(X, m, s, K_lenia):
    U = sp.signal.convolve2d(X, K_lenia, mode='same', boundary='wrap')  # Opération de convolution
    X = np.clip(X + dt * growth_lenia(U, m, s), 0, 1)  # Mettre à jour le motif avec la fonction de croissance et restreindre les valeurs
    return X

# Fonction pour faire évoluer le motif lenia en utilisant une évolution optimisée pour un noyau multi-anneaux
def evolve_hydro(X, m, s, K_lenia):
    fK = np.pad(K_lenia, ((M - len(K_lenia)) // 2, (N - len(K_lenia[0])) // 2), mode='constant')  # Remplir le noyau pour correspondre à la taille du motif
    fK = np.fft.fft2(np.fft.fftshift(fK))  # Effectuer une transformation de Fourier rapide 2D
    U = np.real(np.fft.ifft2(fK * np.fft.fft2(X)))  # Calculer la partie réelle de l'inverse de la FFT du produit du noyau transformé et du motif
    X = np.clip(X + dt * growth_lenia(U, m, s), 0, 1)  # Mettre à jour le motif avec la fonction de croissance et restreindre les valeurs
    return X

# Fonction pour faire évoluer le motif lenia en utilisant une évolution optimisée pour un noyau multi-anneaux avec FFT
def evolve_fft(X, m, s, K_lenia):
    fK = np.pad(K_lenia, ((M - len(K_lenia)) // 2, (N - len(K_lenia[0])) // 2), mode='constant')  # Remplir le noyau pour correspondre à la taille du motif
    fK = np.fft.fft2(np.fft.fftshift(fK))  # Effectuer une transformation de Fourier rapide 2D
    potential_fft = np.fft.fft2(X) * fK  # Calculer la FFT du motif multiplié par le noyau transformé
    potential = np.fft.fftshift(np.real(np.fft.ifft2(potential_fft)))  # Calculer la partie réelle de l'inverse de la FFT décalée du potentiel FFT
    X = np.clip(X + dt * gauss(potential, m, s), 0, 1)  # Mettre à jour le motif avec la fonction de croissance et restreindre les valeurs
    return X

# Fonction pour faire évoluer le motif en utilisant les règles de SmoothLife (non entièrement implémenté)
def smoothlife(X, m, s, K_lenia, B=[0.257, 0.336], D=[0.365, 0.549]):
    ZR = sp.signal.convolve2d(X, K_lenia, mode='same', boundary='wrap')  # Opération de convolution
    birth = np.logical_and(ZR > B[0], ZR < B[1])  # Calculer les conditions de naissance
    death = np.logical_and(ZR > D[0], ZR < D[1])  # Calculer les conditions de décès
    newX = np.where(birth, 1, X)  # Mettre à jour le motif en fonction des conditions de naissance
    newX = np.where(death, 0, newX)  # Mettre à jour le motif en fonction des conditions de décès
    X = np.clip((1 - dt) * X + dt * newX, 0, 1)  # Mettre à jour le motif avec de nouvelles valeurs et restreindre les valeurs
    return X

# Fonction pour faire évoluer le motif en utilisant les règles du jeu de la vie de Conway
def GameOfLife(X, m, s, K_lenia):
    neighbors = sp.signal.convolve2d(X, K_lenia, mode='same', boundary='wrap')  # Opération de convolution pour calculer les voisins
    newGrid = np.where((X == 1) & ((neighbors < 2) | (neighbors > 3)), 0, X)  # Appliquer les règles pour les cellules vivantes
    newGrid = np.where((X == 0) & (neighbors == 3), 1, newGrid)  # Appliquer les règles pour les cellules mortes
    X[:] = newGrid  # Mettre à jour le motif avec la nouvelle grille
    return X

# Fonction pour faire evoluer le motif avec un kernel multiple en utilisant FFT
def evolve_multiple_couples(X, ms=[0.156,0.193,0.342], ss=[0.0118,0.049,0.0891], K_lenia=[]):
    dt = 0.1
    bs = [[1,5/12,2/3],[1/12,1],[1]]
    ms = [0.156,0.193,0.342]
    ss = [0.0118,0.049,0.0891]
    R = 10
    ring_strengths = [0.5, 1, 0.667]
    nb_rings = len(ring_strengths)
    fhs_y = N // 2 
    fhs_x = M // 2
    y, x = np.ogrid[-fhs_y:fhs_y, -fhs_x:fhs_x]
    fKs = []
    for b in bs:
        distance = np.sqrt(x**2 + y**2) / R * len(b)
        K = np.zeros_like(distance)
        mu = 0.5
        sigma = 0.15
        for i in range(len(b)):
    return X

# Nom du fichier

def file_path(name='x', R='x', mu='x', sigma='x', Kernel='x', supp=''):
    file_prefix = "LeniaSimulation"
    file_format = ".gif"
    parameter = f'R{R}mu{mu}sigma{sigma}kernel{Kernel}'
    if supp == '':
        return f"{file_prefix}-{num_file + 1}-{name}-{parameter}{file_format}"
    else:
        return f"{file_prefix}-{num_file + 1}-{name}-{parameter}-{supp}{file_format}"

def produce_movie(X, evolve, m, s, K_lenia, save_path, mode='normal',
                  num_steps=100, cmap=None, interpolation=INTERP):
    if len(X.shape) == 2 and cmap is None:
        cmap = 'gray_r'
    
    start_time = time.time()
    
    convolved_images = []
    deltas = []

    if mode in ['conv', 'delta', 'all']:
        progress_color = {'conv': 'green', 'delta': 'yellow', 'all': 'magenta'}
        progress_bar_preprocessing = tqdm(total=num_steps, desc="Pré-traitement", unit="tab", colour=progress_color.get(mode, 'blue'), leave=False)
        for _ in range(num_steps):
            grid = evolve(X, m, s, K_lenia)
            convolved_grid = sp.signal.convolve2d(grid, K_lenia, mode='same', boundary='wrap')
            convolved_images.append(convolved_grid)
            if mode in ['delta', 'all']:
                delta_grid = evolve(convolved_grid, m, s, K_lenia)
                deltas.append(delta_grid)
            X[:] = np.clip(grid, 0, 1)
            progress_bar_preprocessing.update(1)
        progress_bar_preprocessing.close()

    modes_to_generate = ['normal', 'conv', 'delta'] if mode == 'all' else [mode]
    
    for current_mode in modes_to_generate:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(X, cmap=cmap, interpolation=interpolation, vmin=0, vmax=1)
        ax.xaxis.set_tick_params(rotation=45, labelsize='small')
        ax.yaxis.set_tick_params(rotation=45, labelsize='small')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, extend='both', spacing='proportional')
        cbar.ax.patch.set_alpha(0.6)

        title = f"Simulation de Lenia : {'f(X)' if current_mode == 'normal' else 'k * f(X)' if current_mode == 'conv' else 'd(k * f(X))'} - n°{num_file+1}"
        plt.suptitle(title, x=0.5, y=0.95, ha='center')
        progress_color = {'normal': 'blue', 'conv': 'green', 'delta': 'yellow'}
        progress_bar_video = tqdm(total=num_steps, desc=f"Progression (mode:{current_mode})", unit="img", colour=progress_color[current_mode], leave=False)
        annotations = []  

        def update(i):
            progress_bar_video.update(1)
            
            # Supprimer toutes les annotations existantes
            while annotations:
                annotation = annotations.pop()
                annotation.remove()

            if current_mode == 'normal':
                if i > 0:
                    nonlocal X
                    grid = evolve(X, m, s, K_lenia)
                    im.set_array(grid)
                    X[:] = grid
            elif current_mode == 'conv':
                im.set_array(convolved_images[i])
            elif current_mode == 'delta':
                im.set_array(deltas[i])

            annotation1 = ax.annotate(f"Image {i+1}/{num_steps}", xy=(0.02, 0.96), xycoords='axes fraction',
                                       path_effects=[path_effects.withStroke(linewidth=1, foreground='black')],
                                       fontsize=8, color='white')
            annotation2 = ax.annotate(f"{(i+1)/FPS:.2f}/{num_steps/FPS:.2f} s", xy=(0.02, 0.02), xycoords='axes fraction',
                                       path_effects=[path_effects.withStroke(linewidth=1, foreground='black')],
                                       fontsize=8, color='white')
            annotations.extend([annotation1, annotation2])
    print(f"Temps total pris pour générer les vidéos : {time_taken}")

# Definition des fonctions d'initialisation

# Point gaussien centré sur la grille
def GaussSpot():
    radius = 36
    y, x = np.ogrid[-N//2:N//2, -M//2:M//2]
    grid = gauss3D(x, y, 0, 0, radius, radius)
    return grid

# Pour placer une grille 1 sur une grille 2 de taille N par M
def CenterData(data, flip=False, N=N, M=M):
    grid = np.zeros((N, M))
    start_x = (M - data.shape[1]) // 2
    start_y = (N - data.shape[0]) // 2
    grid[start_y:(start_y + data.shape[0]), start_x:(start_x + data.shape[1])] = data
    if flip:
        grid = np.flipud(grid)
    return grid

# Pour placer une grille 1 dans un angle d'une grille 2 de taille N par M
def PlaceData(data, hor, vert, flip=False, N=N, M=M): 
    grid = np.zeros((N, M))
    # Calcul des positions en fonction des options verticales et horizontales
    if hor == 'left':
        pos_x = int(M * 0.1)
    elif hor == 'right':
        pos_x = int(M * 0.9) - data.shape[1]
    else:
        raise ValueError("Invalid horizontal position")
        
    if vert == 'top':
        pos_y = int(N * 0.1)
    elif vert == 'bottom':
        pos_y = int(N * 0.9) - data.shape[0]
    else:
        raise ValueError("Invalid vertical position")
    grid[pos_y:(pos_y + data.shape[0]), pos_x:(pos_x + data.shape[1])] = data
    
    if flip:
        grid = np.flipud(grid)
        
    return grid

# Pour placer 2 fois la grille 1 sur une grille 2 de taille N par M
def TwoPLace(data, flip=False, N=N, M=M):
    grid = PlaceData(data, 'left', 'top', flip, N, M) + PlaceData(data, 'right', 'bottom', flip, N, M)
    return grid

# Pour placer une grille 1 dans chaque angle d'une grille 2 de taille N par M
def AllPLace(data):
    grid = PlaceData(data, 'top', 'right') + PlaceData(data, 'bottom', 'left') + PlaceData(data, 'top', 'left') + PlaceData(data, 'bottom', 'right')
plt.show()

# Liste des choix
choices = [
        " (1) Gaussian Spot",
        " (2) Orbium Unicaudatus",
        " (3) Random",
        " (4) Random Big Square",
        " (5) Gyropteron Arcus (unworking)",
        " (6) Scutium Solidus",
        " (7) Hydrogeminium Natans",
        " (8) Random with N-rings Kernel",
        " (9) Multiples Rings Gaussian",
        "(10) Random Smalls Squares",
        "(11) Gradient",
        "(12) Small Bug (unworking)",
        "(13) Random Square SmoothLife (unworking)",
        "(14) Compilation d'espèces",
        "(15) Trihelicium Pachus (unworking)",
        "(16) Discutium Pachus (unworking)",
        "(17) Circogeminium Ventilans",
        "(18) Gyrogeminium Serratus",
        "(19) Kernel Aléatoire (unworking)"
        "(20) Triorbium",
        "(21) Decascutium",
        "(22) Catenoscutium Bidirectu",
        "(23) Vagopteron",
        "(24) Heptapteryx Serratus Liquefaciens",
        "(25) Hexacaudopteryx",
        "(26) Catenopteryx Cyclon",
        "(27) Catenohelicium Cyclon Scutoides",
        "(28) Catenohelicium Bispirae Scutoides",
        "(29) Decadentium Volubilis",
        "(30) Aerogeminium Quietus",
        "(31) Hydrogeminium Natans 2",
        "(32) Random - Game of Life",
        "(33) Glider Gun - Game of Life",
        "(34) Weekender - Game of Life",
        "(35) Space Filler - Game of Life",
        "(36) Pufferfish - Game of Life",
        "(37) R-pentomino - Game of Life",
        "(38) Hexastrium",
        "(39) Fish",
        "(40) Random Squares Multiple Couples Kernel",
        "(41) Dodecadentium Nausia",
        "(42) Dodecafolium Ventilans",
]

# Valeurs par défauts, surtout pour créer les variables
choice = 1
frame = 10
choice_conv = 'a'

def asking(choix='', image='', forme=''):
    if choix == '' and image == '' and forme =='':
        # Affichage des choix d'initialisation
        print("Choisissez l'initialisation :")
        for i in choices:
            print(f"  {i}")

        while True:
stop = choice == 'stop' or frame == 'stop' or choice_conv == 'stop'

# Dictionnaire des configurations proposées

EntityInfo = Dict[str, Any]

config: Dict[str, EntityInfo] = {
    '1': {
        'mu': 0.15, 'sigma': 0.017, 'name': 'GaussianSpot', 'R': 26, 'kernel': 1,
        'rings': [], 'evolve': evolve_lenia, 'init': lambda: GaussSpot()
    },
    '2': {
        'mu': 0.15, 'sigma': 0.017, 'name': 'OrbiumUnicaudatus', 'R': 13, 'kernel': 1,
        'rings': [], 'evolve': evolve_lenia, 'init': lambda: TwoPLace(Orbium_unicaudatus)
    },
    '3': {
        'mu': 0.15, 'sigma': 0.017, 'name': 'Random', 'R': 26, 'kernel': 1,
        'rings': [], 'evolve': evolve_lenia, 'init': lambda: RandomGrid()
    },
    '4': {
        'mu': 0.31, 'sigma': 0.049, 'name': 'RandomSquare', 'R': 26, 'kernel': 1,
        'rings': [], 'evolve': evolve_lenia, 'init': lambda: RandomSquare()
    },
    '5': {
        'mu': 0.293, 'sigma': 0.0511, 'name': 'GyropteronArcus', 'R': 26, 'kernel': 1,
        'rings': [], 'evolve': evolve_lenia, 'init': lambda: PlaceData(Gyropteron_arcus, 'left', 'top', True)
    },
    '6': {
        'mu': 0.29, 'sigma': 0.043, 'name': 'ScutiumSolidus', 'R': 26, 'kernel': 1,
        'rings': [], 'evolve': evolve_lenia, 'init': lambda: PlaceData(Scutium_solidus, 'left', 'top')
    },
    '7': {
        'mu': 0.26, 'sigma': 0.036, 'name': 'HydrogeminiumNatans', 'R': 18, 'kernel': 4,
        'rings': [1/2, 1, 2/3], 'evolve': evolve_hydro, 'init': lambda: PlaceData(Hydrogeminium, 'left', 'top')
    },
    '8': {
        'mu': 0.26, 'sigma': 0.036, 'name': 'RandomNrings', 'R': 26, 'kernel': 4,
        'rings': vectn(rd.randint(2, 7)), 'evolve': evolve_hydro, 'init': lambda: RandomSquare()
    },
    '9': {
        'mu': 0.29, 'sigma': 0.027, 'name': 'MutlipleRings', 'R': 26, 'kernel': 4,
        'rings': vectn(3), 'evolve': evolve_hydro, 'init': lambda: GaussRing(0.29, 0.043)
    },
    '10': {
        'mu': 0.15, 'sigma': 0.017, 'name': 'RandomSquares', 'R': 26, 'kernel': 1,
        'rings': [], 'evolve': evolve_lenia, 'init': lambda: RandomSquares(0.15, 0.017, 15)
    },
    '11': {
        'mu': 0.1, 'sigma': 0.15, 'name': 'Gradient', 'R': 26, 'kernel': 1,
        'rings': [], 'evolve': evolve_lenia, 'init': lambda: Gradient()
    },
    '12': {
        'mu': 0.31, 'sigma': 0.048, 'name': 'SmallBug', 'R': 13, 'kernel': 1,
        'rings': [], 'evolve': evolve_lenia, 'init': lambda: PlaceData(Bug, 'left', 'top')
    },
    '13': {
        'mu': 0.31, 'sigma': 0.049, 'name': 'SmoothLifeRandomSquares', 'R': 13, 'kernel': 5,
        'rings': [], 'evolve': smoothlife, 'init': lambda: RandomSquares(0.31, 0.049, 9)
    },
    '14': {
        'mu': 0.337, 'sigma': 0.057, 'name': 'Compilation', 'R': 13, 'kernel': 1,
        'rings': [], 'evolve': evolve_lenia, 'init': lambda: CenterData(Compilation)
    },
    '15': {
        'mu': 0.46, 'sigma': 0.119, 'name': 'TriheliciumPachus', 'R': 13, 'kernel': 3,
        'rings': [], 'evolve': smoothlife, 'init': lambda: PlaceData(TriheliciumPachus, 'left', 'top')
    },
    '16': {
        'mu': 0.545, 'sigma': 0.186, 'name': 'DiscutiumPachus', 'R': 13, 'kernel': 3,
        'rings': [], 'evolve': smoothlife, 'init': lambda: PlaceData(DiscutiumPachus, 'left', 'top')
    },
    '17': {
        'mu': 0.29, 'sigma': 0.035, 'name': 'CircogeminiumVentilans', 'R': 45, 'kernel': 4,
        'rings': [1, 1, 1], 'evolve': evolve_hydro, 'init': lambda: CenterData(BigCircogeminiumVentilans)
    },
    '18': {
        'mu': 0.27, 'sigma': 0.04, 'name': 'GyrogeminiumSerratus', 'R': 36, 'kernel': 4,
        'rings': [1/2, 1, 1/2], 'evolve': evolve_hydro, 'init': lambda: CenterData(GyrogeminiumSerratus)
    },
    '19': {
        'mu': round(np.random.uniform(0.1, 0.5), 3), 'sigma': round(np.random.uniform(0.01, 0.18), 3), 'name': 'KernelAleatoire',
}

def simulation(choix='', image='', forme=''): 
    """
    choix : choix de l'initialisation
    image :  nombre d'image(s) à génerer
    forme : choix de la visualisation
    """
    asking(choix, image, forme) # dialogue de demande

    global frame, mu, frame, rings, X, k_choice, num_file, tab, stop, CMAP, dt, DPI, total_time
    
    if choice in config: # attitrage des variables
        cfg = config[choice]
        mu, sigma = cfg['mu'], cfg['sigma']
        R = cfg['R']
        name = cfg['name']
        X = cfg['init']()
        k_choice = cfg['kernel']
        rings = cfg['rings']
        K_lenia = Kernel_choice(k_choice, rings, R)
        evolve_func = cfg['evolve']
        
        if choice in ['32', '33', '34', '35', '36', '37']:
            interpolation = 'none'
            interpolation2 = 'none'
        else:
            interpolation = INTERP
            interpolation2 = INTERP2

        # Affichage d'une figure avec le kernel l'initialisation et les paramètres de simulation
        fig = plt.figure(figsize=(12, 4.2))
        fig.suptitle("Choix d'initialisation et de Simulation", fontsize=16, x=0.45, y=0.95)
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.06, 1])

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        cbar_ax = fig.add_subplot(gs[0, 2])
        ax_info = fig.add_subplot(gs[0, 3])

        im0 = ax0.imshow(K_lenia, cmap=CMAP, interpolation=interpolation2)
        ax0.set_title(f'Kernel choisi: K{k_choice}')
        ax0.xaxis.set_tick_params(rotation=45, labelsize='small')
        ax0.set_yticks([])

        if choice_conv == 'b':
            affichage = ' Convolution : $k * f(X)$'
            SHOW = np.clip(sp.signal.convolve2d(X, K_lenia, mode='full', boundary='wrap'), 0, 1)
        elif choice_conv == 'c':
            affichage = ' Delta : $d(k * f(X))$'
            SHOW = np.clip(evolve_func(sp.signal.convolve2d(X, K_lenia, mode='same', boundary='wrap'), mu, sigma, K_lenia), 0, 1)
        elif choice_conv == 'a':
            affichage = ' Normal : $f(X)$'
            SHOW = X
        elif choice_conv == 'd':
            affichage = ' Tous les affichages ont été choisis.\n $f(X)$ affiché par défaut.'
            SHOW = X
        else :
            affichage ='error'
            SHOW = X

        im1 = ax1.imshow(SHOW, cmap=CMAP, interpolation=interpolation, vmin=0, vmax=1)
        ax1.set_title('Initialisation choisie')
        ax1.set_yticks([])
        ax1.xaxis.set_tick_params(rotation=45, labelsize='small')

        cbar = fig.colorbar(im1, cax=cbar_ax, shrink=0.7, extend='both')
        cbar.ax.set_ylabel('Intensité')

        # Paramètres de simulation (text)
        info_text = (
            f"Simulation n°{num_file + 1}\n\n"
            f"Nom de l'initialisation choisie :\n"
            f" {unclean(name)}\n \n"
            f"mu = {mu} ; sigma = {sigma} ; dt = {dt}\n"
            f"R = {R} ; rings = {np.round(rings,2)} \n \n"
            f"Numéro de Kernel : K{k_choice}\n \n"
            f"Nombre d'images : {frame}\n \n"
            f"Taille du graphe : [{np.shape(X)[0]}:{np.shape(X)[1]}]\n"
            f"DPI = {DPI} ; fps = {FPS}\n \n"
            f"ColorMap: {get_colormap_name(CMAP)}\n\n"
            f"Type d'affichage : \n"
            f"{affichage}"
                    )

        ax_info.text(0.05, 0.5, info_text, va='center', ha='left', fontsize=11)
        ax_info.axis('off')  # Pour enlever les axes du subplot d'info

        plt.tight_layout()
        plt.show()

        # Génération de la vidéo
        if frame != 0:
            if tab[0] is not None:
                print(f"\nDémarrage de la simulation...\n\nIl faut compter environ {tab[0]} s par images.")
                temps = np.round(frame*tab[0] if choice_conv != 'd' else frame*5*tab[0],2)
                if temps > 60:
                    print(f"Le chargement devrait prendre environ {np.round(temps/60,2)} minutes.")
                else:
                    print(f"Le chargement devrait prendre environ {np.round(temps,2)} secondes.")
                print("Ces valeurs sont basées sur les temps des dernières simulations.\nElles peuvent être peu fiables car de nombreux facteurs peuvent varier d'une simulation à l'autre.")
            
            if choice_conv == 'a': # f(x)
                produce_movie(X, evolve_func, mu, sigma, K_lenia, file_path(name, R, mu, sigma, k_choice, 'normal'), mode='normal', num_steps=frame, cmap=CMAP, interpolation=interpolation)
            elif choice_conv == 'b': # k * f(x)
        print(" -> Erreur dans lors du choix de l'initialisation")

simulation()