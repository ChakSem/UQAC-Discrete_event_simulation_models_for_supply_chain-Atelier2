# 🚀 Supply Chain Optimization Demo

## 📋 Description

Cette démonstration présente une approche complète d'optimisation d'une chaîne d'approvisionnement simplifiée en utilisant la simulation et différents algorithmes d'optimisation.

### 🎯 Objectifs
- Modéliser une chaîne d'approvisionnement simple (1 grossiste → 1 détaillant)
- Comparer 5 algorithmes d'optimisation différents
- Identifier la configuration optimale des stocks (politique S,s)
- Maximiser le profit net quotidien

## 📁 Structure du projet

```
supply_chain_optimization_demo/
│
├── 📊 notebooks/                        # Notebooks Jupyter
│   ├── 01_introduction_setup.ipynb     # Configuration et introduction
│   ├── 02_simulation_base.ipynb        # Modèle de simulation SimPy
│   ├── 03_exploration_visualisation.ipynb # Exploration de l'espace
│   ├── 04_optimisation_comparaison.ipynb  # Comparaison des optimiseurs
│   └── 05_resultats_conclusions.ipynb     # Synthèse et recommandations
│
├── 📈 src/                             # Code source
│   └── supply_chain_simple.py          # Module de simulation
│
├── 📊 results/                         # Résultats générés
│   ├── figures/                        # Graphiques
│   └── data/                          # Données sauvegardées
│
└── README.md                          # Ce fichier
```

## 🚀 Démarrage rapide

### 1. Installation des dépendances

```bash
pip install numpy pandas matplotlib seaborn simpy scipy scikit-learn plotly ipywidgets
```

### 2. Exécution séquentielle

Exécuter les notebooks dans l'ordre :

1. **01_introduction_setup.ipynb** : Configuration initiale
2. **02_simulation_base.ipynb** : Création du modèle
3. **03_exploration_visualisation.ipynb** : Analyse exploratoire
4. **04_optimisation_comparaison.ipynb** : Optimisation
5. **05_resultats_conclusions.ipynb** : Synthèse finale

## 📊 Cas d'étude

### Chaîne simplifiée
```
Fabricant (∞) → Grossiste (S,s) → Détaillant (S,s) → Clients
```

### Variables de décision
- `S_grossiste` : Capacité maximale du grossiste
- `s_grossiste` : Seuil de réapprovisionnement du grossiste  
- `S_detaillant` : Capacité maximale du détaillant
- `s_detaillant` : Seuil de réapprovisionnement du détaillant

### Objectif
Maximiser le profit net quotidien = Revenus - (Coûts stockage + Coûts livraison)

## 🔧 Algorithmes comparés

1. **Monte Carlo** : Recherche aléatoire simple
2. **Grid Search** : Recherche exhaustive sur grille
3. **GPR Bayesian** : Optimisation bayésienne avec métamodèle
4. **Differential Evolution** : Algorithme évolutionnaire
5. **Hybrid (GPR + Local)** : Approche combinée

## 📈 Résultats clés

### Performance optimale
- **Profit** : ~2700-2800€/jour
- **Configuration** : S_grossiste ≈ 900, s_grossiste ≈ 350, S_détaillant ≈ 200, s_détaillant ≈ 65
- **Taux de service** : >95%

### Comparaison des algorithmes

| Algorithme | Profit (€/j) | Temps (s) | Évaluations | Recommandé pour |
|------------|--------------|-----------|-------------|-----------------|
| Monte Carlo | 2650 | 10 | 500 | Exploration initiale |
| Grid Search | 2700 | 45 | 625 | Petits espaces |
| **GPR Bayesian** | **2780** | **30** | **50** | **Optimisation coûteuse** ⭐ |
| Diff. Evolution | 2750 | 25 | 200 | Robustesse |
| Hybrid | 2800 | 40 | 70 | Performance max |

## 💡 Insights principaux

1. **GPR est le meilleur compromis** efficacité/performance
2. **50 simulations suffisent** avec un bon métamodèle
3. **Configuration équilibrée** : gros stock grossiste, stock modéré détaillant
4. **ROI rapide** : rentabilité en moins de 2 mois

## 🎯 Utilisation pratique

### Pour votre propre chaîne

1. **Adapter le modèle** dans `02_simulation_base.ipynb`
2. **Modifier les paramètres** dans `01_introduction_setup.ipynb`
3. **Lancer l'optimisation** avec `04_optimisation_comparaison.ipynb`
4. **Analyser les résultats** dans `05_resultats_conclusions.ipynb`

### Extensions possibles

- Multi-produits
- Multi-échelons
- Demande stochastique complexe
- Optimisation multi-objectifs
- Contraintes capacitaires

## 📚 Références

- **SimPy** : Framework de simulation discrete
- **GPR** : Gaussian Process Regression (scikit-learn)
- **Supply Chain** : Politique (S,s) classique
