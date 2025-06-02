# Démarche de Réimplémentation - Optimisation de Chaîne d'Approvisionnement

## 📌 Vue d'ensemble
Réimplémentation de l'article *An Open Tool-Set for Simulation, Design-Space Exploration and Optimization of Supply Chains* avec SimPy.

---

## Phase 1 : Compréhension et Préparation (Semaine 1)

### 📖 Lecture approfondie
- [ ] Lire l'article complet en notant les concepts clés
- [ ] Identifier les composants principaux du système
- [ ] Comprendre la politique d'inventaire (S,s)
- [ ] Noter les métriques de performance utilisées

### 🔧 Configuration de l'environnement
- [ ] Installer Python 3.8+
- [ ] Créer un environnement virtuel :  
    ```bash
    python -m venv venv
    ```
- [ ] Installer les dépendances :
    ```bash
    pip install simpy numpy pandas matplotlib scipy
    pip install jupyterlab seaborn
    ```
- [ ] Tester SimPy avec un exemple simple

### 📊 Analyse du problème
- [ ] Schématiser la chaîne d'approvisionnement
- [ ] Lister les paramètres fixes vs variables
- [ ] Identifier les simplifications possibles
- [ ] Documenter les hypothèses

---

## Phase 2 : Implémentation de Base (Semaine 2)

### 🏗️ Architecture du simulateur
- [ ] Créer une classe `SupplyChainNode` de base
- [ ] Implémenter un `Container` pour l'inventaire
- [ ] Créer un `Monitor` pour la surveillance
- [ ] Développer la logique de commande automatique

### 👥 Composants spécifiques
- [ ] Implémenter la classe `Manufacturer` (capacité illimitée)
- [ ] Implémenter la classe `Distributor` avec inventaire
- [ ] Implémenter la classe `Retailer` avec inventaire
- [ ] Créer un générateur d'arrivées clients (Poisson)

### ✅ Tests unitaires
- [ ] Tester chaque composant individuellement
- [ ] Vérifier la logique de réapprovisionnement
- [ ] Valider les calculs de coûts

---

## Phase 3 : Simulation Complète (Semaine 3)

### 🔄 Intégration
- [ ] Connecter tous les composants
- [ ] Implémenter la sélection intelligente du fournisseur
- [ ] Ajouter la collecte des métriques
- [ ] Créer la fonction `single_sim_run()`

### 📈 Validation
- [ ] Comparer avec les résultats de l'article
- [ ] Analyser la cohérence des tendances
- [ ] Documenter les écarts observés

### ⚡ Optimisations (si PC limité)
- [ ] Réduire la durée de simulation (360 jours → 100 jours)
- [ ] Diminuer le nombre de runs (200 → 50)
- [ ] Simplifier le réseau (2x2 → 1x2)
- [ ] Utiliser l'échantillonnage adaptatif

---

## Phase 4 : Analyse et Visualisation (Semaine 4)

### 📊 Analyse coût-précision
- [ ] Implémenter le calcul RSE (Relative Standard Error)
- [ ] Créer un graphique RSE vs nombre de simulations
- [ ] Déterminer le point optimal simulations/précision
- [ ] Documenter les compromis temps/précision

### 🎨 Visualisations clés
- [ ] Graphique profit net vs paramètres (s,S)
- [ ] Évolution des niveaux d'inventaire
- [ ] Taux de clients perdus
- [ ] Surface 3D d'optimisation

### 📝 Documentation
- [ ] Créer un notebook explicatif
- [ ] Ajouter du markdown entre les cellules
- [ ] Inclure des interprétations des résultats
- [ ] Préparer une démo interactive

---

## Phase 5 : Optimisation (Optionnel)

### 🔍 Exploration simplifiée
- [ ] Réduire l'espace de recherche (4 paramètres au lieu de 8)
- [ ] Utiliser une grille grossière (3x3 au lieu de 4x4)
- [ ] Implémenter une recherche locale simple
- [ ] Comparer avec les résultats de l'article

### 🎯 Méta-modèle (si temps)
- [ ] Créer un modèle polynomial simple
- [ ] Tester sur un sous-ensemble de points
- [ ] Visualiser l'approximation
- [ ] Discuter les limitations

---

## 📋 Checklist finale

### Livrables obligatoires
- [ ] Code source commenté (`.ipynb`)
- [ ] Résultats reproductibles
- [ ] Comparaison avec l'article original
- [ ] Présentation des écarts et justifications

### Documentation
- [ ] README avec instructions
- [ ] Explication des simplifications
- [ ] Analyse de complexité computationnelle
- [ ] Suggestions d'améliorations

### Démonstration
- [ ] Notebook exécutable de bout en bout
- [ ] Temps d'exécution < 30 minutes
- [ ] Visualisations interactives
- [ ] Conclusions claires

---

## 🚨 Points d'attention

- **Performance** : Si exécution > 1h, réduire davantage les paramètres
- **Mémoire** : Surveiller l'usage RAM, limiter l'historique si nécessaire
- **Validation** : Toujours comparer les tendances, pas les valeurs absolues
- **Documentation** : Expliquer POURQUOI chaque simplification

---

## 💡 Conseils

- Commencer simple et complexifier progressivement
- Sauvegarder les résultats intermédiaires (`pickle`/CSV)
- Utiliser un profiler Python si problèmes de performance
- Demander un feedback régulier au professeur