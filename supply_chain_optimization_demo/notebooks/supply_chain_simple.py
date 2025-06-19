"""
Module de simulation de chaîne d'approvisionnement simplifiée
=============================================================

Ce module fournit les classes et fonctions nécessaires pour simuler
une chaîne d'approvisionnement avec politique de stock (S,s).

Auteur: Supply Chain Optimization Demo
Date: 2024
"""

import simpy
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class SimulationResults:
    """Résultats d'une simulation"""
    profit_net: float
    taux_service: float
    stock_moyen_grossiste: float
    stock_moyen_detaillant: float
    cout_stockage_total: float
    cout_livraison_total: float
    ventes_totales: int
    clients_perdus: int
    rupture_grossiste: float = 0.0
    rupture_detaillant: float = 0.0
    
    def __str__(self):
        return f"""
        📊 Résultats de simulation:
        - Profit net: {self.profit_net:.2f}€/jour
        - Taux de service: {self.taux_service:.1%}
        - Stock moyen (G/D): {self.stock_moyen_grossiste:.0f}/{self.stock_moyen_detaillant:.0f}
        - Ventes totales: {self.ventes_totales}
        """


class SupplyChainNode:
    """Classe de base pour un nœud de la chaîne d'approvisionnement"""
    
    def __init__(self, env: simpy.Environment, name: str, 
                 S: Optional[int] = None, s: Optional[int] = None,
                 holding_cost: float = 0):
        self.env = env
        self.name = name
        self.S = S  # Capacité maximale
        self.s = s  # Seuil de réapprovisionnement
        self.holding_cost = holding_cost
        
        # Stock (None pour capacité infinie)
        if S is not None:
            self.stock = simpy.Container(env, capacity=S, init=S)
        else:
            self.stock = None
            
        # Statistiques
        self.total_sold = 0
        self.total_ordered = 0
        self.stock_levels = []
        self.stock_times = []
        self.order_history = []
        self.stockout_time = 0
        self.last_level = S if S else float('inf')
        self.last_time = 0
        
    def record_stock_level(self):
        """Enregistre le niveau de stock actuel"""
        current_time = self.env.now
        current_level = self.stock.level if self.stock else float('inf')
        
        # Enregistrer le temps de rupture
        if self.last_level == 0:
            self.stockout_time += (current_time - self.last_time)
            
        self.stock_levels.append(current_level)
        self.stock_times.append(current_time)
        self.last_level = current_level
        self.last_time = current_time
        
    def get_average_stock(self):
        """Calcule le stock moyen pondéré par le temps"""
        if len(self.stock_levels) < 2:
            return self.S if self.S else 0
            
        total_weighted = 0
        total_time = 0
        
        for i in range(1, len(self.stock_times)):
            dt = self.stock_times[i] - self.stock_times[i-1]
            total_weighted += self.stock_levels[i-1] * dt
            total_time += dt
            
        return total_weighted / total_time if total_time > 0 else 0
    
    def get_stockout_percentage(self):
        """Pourcentage du temps en rupture de stock"""
        if self.env.now == 0:
            return 0
        return (self.stockout_time / self.env.now) * 100


class Retailer(SupplyChainNode):
    """Détaillant avec politique (S,s)"""
    
    def __init__(self, env, name, S, s, holding_cost, supplier, 
                 order_cost, lead_time):
        super().__init__(env, name, S, s, holding_cost)
        self.supplier = supplier
        self.order_cost = order_cost
        self.lead_time = lead_time
        self.ordering = False
        self.customers_served = 0
        self.customers_lost = 0
        self.total_demand = 0
        
        # Lancer le processus de surveillance du stock
        env.process(self.monitor_stock())
        
    def monitor_stock(self):
        """Surveille le stock et passe commande si nécessaire"""
        while True:
            yield self.env.timeout(1)  # Vérifier chaque jour
            self.record_stock_level()
            
            if self.stock.level <= self.s and not self.ordering:
                self.env.process(self.order_from_supplier())
                
    def order_from_supplier(self):
        """Passe commande au fournisseur"""
        self.ordering = True
        order_quantity = self.S - self.stock.level
        
        # Vérifier la disponibilité chez le fournisseur
        if self.supplier.stock and self.supplier.stock.level >= order_quantity:
            # Retirer du stock fournisseur
            yield self.supplier.stock.get(order_quantity)
            self.supplier.total_sold += order_quantity
            
            # Attendre le délai de livraison
            yield self.env.timeout(self.lead_time)
            
            # Recevoir la marchandise
            yield self.stock.put(order_quantity)
            self.total_ordered += order_quantity
            self.order_history.append({
                'time': self.env.now,
                'quantity': order_quantity,
                'cost': self.order_cost
            })
        
        self.ordering = False
        
    def sell_to_customer(self, quantity):
        """Vend au client si stock disponible"""
        self.total_demand += quantity
        
        if self.stock.level >= quantity:
            yield self.stock.get(quantity)
            self.total_sold += quantity
            self.customers_served += 1
            return True
        else:
            self.customers_lost += 1
            return False


class Wholesaler(SupplyChainNode):
    """Grossiste avec politique (S,s)"""
    
    def __init__(self, env, name, S, s, holding_cost, manufacturer,
                 order_cost, lead_time):
        super().__init__(env, name, S, s, holding_cost)
        self.manufacturer = manufacturer
        self.order_cost = order_cost
        self.lead_time = lead_time
        self.ordering = False
        
        # Lancer le processus de surveillance
        env.process(self.monitor_stock())
        
    def monitor_stock(self):
        """Surveille le stock et commande si nécessaire"""
        while True:
            yield self.env.timeout(1)
            self.record_stock_level()
            
            if self.stock.level <= self.s and not self.ordering:
                self.env.process(self.order_from_manufacturer())
                
    def order_from_manufacturer(self):
        """Commande au fabricant"""
        self.ordering = True
        order_quantity = self.S - self.stock.level
        
        # Le fabricant a une capacité infinie
        yield self.env.timeout(self.lead_time)
        yield self.stock.put(order_quantity)
        
        self.total_ordered += order_quantity
        self.order_history.append({
            'time': self.env.now,
            'quantity': order_quantity,
            'cost': self.order_cost
        })
        
        self.ordering = False


def customer_generator(env, retailer, arrival_rate, min_demand, max_demand):
    """Génère des clients selon un processus de Poisson"""
    customer_id = 0
    
    while True:
        # Temps jusqu'au prochain client (exponentiel)
        yield env.timeout(np.random.exponential(1/arrival_rate))
        
        customer_id += 1
        demand = np.random.randint(min_demand, max_demand + 1)
        
        # Le client essaie d'acheter
        env.process(customer_purchase(env, f"Client_{customer_id}", 
                                    retailer, demand))


def customer_purchase(env, customer_id, retailer, quantity):
    """Processus d'achat d'un client"""
    success = yield env.process(retailer.sell_to_customer(quantity))
    return success


def run_simulation(params: Dict[str, int], config: Dict, verbose: bool = False) -> Tuple[SimulationResults, Retailer, Wholesaler]:
    """
    Execute une simulation avec les paramètres donnés
    
    Args:
        params: Dictionnaire avec S_grossiste, s_grossiste, S_detaillant, s_detaillant
        config: Configuration globale
        verbose: Afficher les logs détaillés
    
    Returns:
        Tuple (SimulationResults, retailer, wholesaler)
    """
    # Créer l'environnement
    env = simpy.Environment()
    
    # Créer le fabricant (capacité infinie)
    manufacturer = SupplyChainNode(env, "Fabricant")
    
    # Créer le grossiste
    wholesaler = Wholesaler(
        env, "Grossiste",
        S=int(params['S_grossiste']),
        s=int(params['s_grossiste']),
        holding_cost=config['cout_stockage_grossiste'],
        manufacturer=manufacturer,
        order_cost=config['cout_livraison_fabricant'],
        lead_time=config['delai_livraison_fabricant']
    )
    
    # Créer le détaillant
    retailer = Retailer(
        env, "Détaillant",
        S=int(params['S_detaillant']),
        s=int(params['s_detaillant']),
        holding_cost=config['cout_stockage_detaillant'],
        supplier=wholesaler,
        order_cost=config['cout_livraison_grossiste'],
        lead_time=config['delai_livraison_grossiste']
    )
    
    # Lancer le générateur de clients
    env.process(customer_generator(
        env, retailer,
        arrival_rate=config['taux_arrivee_clients'],
        min_demand=config['achat_min'],
        max_demand=config['achat_max']
    ))
    
    # Exécuter la simulation
    env.run(until=config['duree_simulation'])
    
    # Calculer les métriques
    # Revenus
    revenue = retailer.total_sold * config['marge_smartphone']
    
    # Coûts de stockage
    holding_cost_wholesaler = wholesaler.get_average_stock() * config['cout_stockage_grossiste'] * config['duree_simulation']
    holding_cost_retailer = retailer.get_average_stock() * config['cout_stockage_detaillant'] * config['duree_simulation']
    total_holding_cost = holding_cost_wholesaler + holding_cost_retailer
    
    # Coûts de livraison
    delivery_cost_wholesaler = sum(order['cost'] for order in wholesaler.order_history)
    delivery_cost_retailer = sum(order['cost'] for order in retailer.order_history)
    total_delivery_cost = delivery_cost_wholesaler + delivery_cost_retailer
    
    # Profit net quotidien
    total_cost = total_holding_cost + total_delivery_cost
    net_profit = revenue - total_cost
    daily_profit = net_profit / config['duree_simulation']
    
    # Taux de service
    total_customers = retailer.customers_served + retailer.customers_lost
    service_rate = retailer.customers_served / total_customers if total_customers > 0 else 0
    
    # Créer les résultats
    results = SimulationResults(
        profit_net=daily_profit,
        taux_service=service_rate,
        stock_moyen_grossiste=wholesaler.get_average_stock(),
        stock_moyen_detaillant=retailer.get_average_stock(),
        cout_stockage_total=total_holding_cost / config['duree_simulation'],
        cout_livraison_total=total_delivery_cost / config['duree_simulation'],
        ventes_totales=retailer.total_sold,
        clients_perdus=retailer.customers_lost,
        rupture_grossiste=wholesaler.get_stockout_percentage(),
        rupture_detaillant=retailer.get_stockout_percentage()
    )
    
    return results, retailer, wholesaler


def run_multiple_simulations(params: Dict, config: Dict, n_replications: int = 30):
    """Execute plusieurs simulations et retourne les statistiques"""
    results_list = []
    
    for i in range(n_replications):
        # Changer la graine aléatoire
        np.random.seed(42 + i)
        random.seed(42 + i)
        
        results, _, _ = run_simulation(params, config, verbose=False)
        results_list.append(results)
    
    # Calculer les moyennes et écarts-types
    metrics = {
        'profit_net': [r.profit_net for r in results_list],
        'taux_service': [r.taux_service for r in results_list],
        'stock_moyen_grossiste': [r.stock_moyen_grossiste for r in results_list],
        'stock_moyen_detaillant': [r.stock_moyen_detaillant for r in results_list],
        'rupture_detaillant': [r.rupture_detaillant for r in results_list]
    }
    
    stats = {}
    for metric, values in metrics.items():
        stats[f'{metric}_mean'] = np.mean(values)
        stats[f'{metric}_std'] = np.std(values)
        stats[f'{metric}_cv'] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
    
    return stats, metrics


def evaluate_supply_chain(params: Dict, config: Dict, n_replications: int = 1) -> float:
    """
    Fonction d'évaluation pour l'optimisation
    
    Args:
        params: Paramètres de la chaîne
        config: Configuration
        n_replications: Nombre de réplications
        
    Returns:
        Profit net moyen (négatif si échec)
    """
    # Vérifier les contraintes
    if params['s_grossiste'] >= params['S_grossiste'] or \
       params['s_detaillant'] >= params['S_detaillant']:
        return -1e6  # Pénalité pour violation de contrainte
    
    if n_replications == 1:
        results, _, _ = run_simulation(params, config, verbose=False)
        return results.profit_net
    else:
        profits = []
        for i in range(n_replications):
            np.random.seed(42 + i)
            random.seed(42 + i)
            results, _, _ = run_simulation(params, config, verbose=False)
            profits.append(results.profit_net)
        return np.mean(profits)


if __name__ == "__main__":
    # Test du module
    print("Test du module de simulation...")
    
    config = {
        'marge_smartphone': 80,
        'cout_stockage_grossiste': 2.0,
        'cout_stockage_detaillant': 5.0,
        'cout_livraison_fabricant': 800,
        'delai_livraison_fabricant': 5,
        'cout_livraison_grossiste': 50,
        'delai_livraison_grossiste': 1,
        'taux_arrivee_clients': 15,
        'achat_min': 1,
        'achat_max': 3,
        'duree_simulation': 90,
    }
    
    params = {
        'S_grossiste': 900,
        's_grossiste': 300,
        'S_detaillant': 200,
        's_detaillant': 50
    }
    
    results, _, _ = run_simulation(params, config)
    print(results)