# Projet Apprentissage par Renforcement : FrozenLake

Ce projet compare deux algorithmes d'apprentissage par renforcement (Q-Learning tabulaire et Deep Q-Network via Stable-Baselines3) sur l'environnement FrozenLakeV1 de Gymnasium. Il a pour but d'analyser les performances de méthodes face à des environnements stochastiques et des récompenses rares.

## Prérequis

* Python 3.8 ou supérieur.

## Installation et Configuration

1. **Créer l'environnement virtuel :**

```bash
python -m venv venv
```

2. **Activer l'environnement virtuel :**

Sous Windows :

```bash
.\venv\Scripts\activate
```

Sous macOS / Linux :

```bash
source venv/bin/activate
```

3. **Installer les dépendances :**

```bash
pip install -r requirements.txt
```

## Utilisation

Le point d'entrée principal du projet est le fichier `main.py`. 

Avant de lancer l'entraînement, vous pouvez configurer l'expérience très simplement en modifiant les variables globales au début du fichier `main.py` :

### 1. Choix de l'environnement et de l'agent

- **`CASE`** : Définit le cas d'étude (la carte) à utiliser.
  - `"1"` : Grille 4x4 basique (glace non glissante / déterministe)
  - `"2"` : Grille 4x4 standard (glace glissante / stochastique)
  - `"3"` : Grille 8x8 personnalisée (glace glissante avec carte complexe)

- **`AGENT`** : Algorithme à entraîner  
  - `"Q-Learning"`  
  - `"DQN"`

- **`N_RUNS`** : Nombre d'exécutions indépendantes pour moyenner les résultats

---

### 2. Hyperparamètres

Les hyperparamètres sont situés dans `main.py` via les dictionnaires `Q_PARAMS` et `DQN_PARAMS`.

Vous pouvez notamment ajuster :
- le taux d'apprentissage
- le taux d'actualisation (`gamma`)
- les différents paramètres d'exploration

---

### 3. Exécution

Depuis la racine du projet :

    python main.py

---

### Résultats générés

À la fin de l'exécution, plusieurs sorties sont produites automatiquement :

- Affichage des métriques dans la console  

- Graphique du taux de réussite avec intervalle de confiance (image sauvegardée dans `results/`)

- Visualisation de la politique apprise (actions optimales par case)

- Simulation de l'agent dans l'environnement
