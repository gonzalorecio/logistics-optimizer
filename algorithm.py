import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from collections import Counter
from random import shuffle

routes = None
costs = None
shops = None
capacity = None
stations_capacity = None
stations = None
num_stations = None
last_optimization_history = None
weights = (1.1, 0, 1)

def set_globals(data, _num_stations):
    global routes, costs, shops, capacity, stations_capacity, stations, num_stations
    num_stations = _num_stations
    capacity = len(data[['Tienda']].values)//num_stations * 1.05
    stations_capacity = [capacity]*num_stations
    stations = [[] for _ in range(num_stations)]
    costs = {t:q for t,q in data[['Tienda', 'Q']].values}
    routes = {t:r for t,r in data[['Tienda', 'Ruta']].values}
    shops = list(data['Tienda'].values)



def get_station_cost(station):
    return sum([costs[shop] for shop in station])

def get_station_routes(station):
    return set([routes[shop] for shop in station])

def get_station_routes_list(station):
    return [routes[shop] for shop in station]

def get_station_load(station):
    return len(station)

def print_solution(stations):
    Q = 0
    for i, station in enumerate(stations):
        capacity = get_station_load(station)
        station_cost = get_station_cost(station)
        station_routes = get_station_routes_list(station)
        print(f'Station {i+1} ({capacity} shops), sum-Q = {station_cost:.4f}, routes: {Counter(station_routes)}')
        Q += station_cost

def print_solution_str(stations):
    Q = 0
    result_str = ''
    for i, station in enumerate(stations):
        capacity = get_station_load(station)
        station_cost = get_station_cost(station)
        station_routes = get_station_routes_list(station)
        result_str += f'Station {i+1} ({capacity} shops), sum-Q = {station_cost:.4f}, routes: {Counter(station_routes)}\n'
        Q += station_cost
    return result_str

def get_best_station_for_shop(stations, shop, return_cost=False):
    all_station_costs = []
    all_station_routes = []
    for i, station in enumerate(stations):
        station_cost = get_station_cost(station)
        station_routes = Counter(get_station_routes_list(station))
        count_routes = station_routes[routes[shop]]
        routes_score = -(count_routes / (sum(station_routes.values()) + 1)) 

        all_station_costs.append(station_cost)
        all_station_routes.append(routes_score)

    scores = np.array(all_station_costs)/5 + (np.array(all_station_routes) * 100)
    if return_cost:
        return np.argmin(scores), min(scores)
    return np.argmin(scores)





"""Metaheuristic functions"""

def rand(min_val, max_val):
    return np.random.randint(min_val, max_val)

def check_is_valid_solution(stations):
    concat_stations = np.array(list(itertools.chain(*stations)))
    return len(concat_stations) == len(shops) and \
           all(np.array(sorted(concat_stations)) == np.array(sorted(shops)))

def get_solution_dispersion(stations):
    station_routes = [get_station_routes_list(s) for s in stations]
    # print(station_routes)
    dispersion = 0
    route_dispersion = 0
    for i, sr1 in enumerate(station_routes):
        for j, sr2 in enumerate(station_routes): 
            if i != j:
                overlapped = set(sr1)&set(sr2)
                dispersion = len(overlapped)
                o1 = [s for s in sr1 if s in sr2]
                o2 = [s for s in sr2 if s in sr1]
                route_dispersion += min(len(o1), len(o2))
    return route_dispersion

def solution_cost(stations, weights=None, verbose=False):
    a, b, c = 1, 0, 5
    exp_a = 1.15#.525
    exp_b = 1.1#2#.525
    if weights:
        a, b, c = weights
    all_costs = np.array([get_station_cost(s) for s in stations])
    expected_cost_q = sum(all_costs)/num_stations
    cost_q = ((np.sqrt((all_costs-expected_cost_q)**2)**exp_a).mean())

    all_capacities = np.array([get_station_load(s) for s in stations])
    expected_balance = sum(all_capacities)/num_stations
    cost_balance = ((all_capacities-expected_balance)**2).mean()

    route_dispersion = get_solution_dispersion(stations)
    # total_cost = a*cost_q + (b*cost_balance+1) * (c*(route_dispersion**2)+1)
    total_cost = (a*cost_q + 1) * (c*(route_dispersion**exp_b)+1)
    total_cost = (a*cost_q + 1) + (c*(route_dispersion**exp_b)+1)
    if verbose:
        print('\t', f'{cost_q**(1/exp_a):.2f}', int(cost_balance), int(route_dispersion))
        print('\t', f'{a*cost_q:.2f}', int(b*cost_balance), int(c*route_dispersion**exp_b), f'Total: {total_cost:.2f}')
    return total_cost

def swap_shops(stations, i, j):
    s1 = stations[i]
    s2 = stations[j]
    if len(s1) == 0 or len(s2) == 0:
        return stations
    k1 = rand(0, len(s1))
    k2 = rand(0, len(s2))
    s1[k1], s2[k2] = s2[k2], s1[k1]
    return stations

def three_swap_shops(stations, i, j, k):
    if len(stations) < 3:
        return stations
    s1 = stations[i]
    s2 = stations[j]
    s3 = stations[k]
    if len(s1) == 0 or len(s2) == 0 or len(s3) == 0 or len(set([i,j,k])) < 3:
        return stations
    x1, x2, x3 = rand(0, len(s1)), rand(0, len(s2)), rand(0, len(s3))
    p2, p3, p1 = copy(s2[x2]), copy(s3[x3]), copy(s1[x1])
    s1[x1]= p2
    s2[x2] = p3
    s3[x3] = p1
    return stations
    
def move_shop(stations, i, j):
    s1 = stations[i]
    s2 = stations[j]
    if len(s1) == 0:
        return stations
    # k = np.argmin([get_best_station_for_shop(stations, s, return_cost=True)[1] for s in s1])
    k = rand(0, len(s1))
    shop = s1[k]
    del s1[k]
    s2.append(shop)
    return stations



def find_initial_solution():
    min_sol_cost = 100000000000
    min_solution = None
    for k in range(750):
        stations = [[] for _ in range(num_stations)]
        shops_shuffled = deepcopy(shops)
        shuffle(shops_shuffled)
        for shop in shops_shuffled:
            i = get_best_station_for_shop(stations, shop)
            stations[i].append(shop) # Insert shop to current station
            
        assert list(sorted(list(itertools.chain(*stations)))) == list(sorted(shops))
        # print_solution(stations)
        cost = solution_cost(stations, weights=weights)
        if cost < min_sol_cost:
            print(k, cost)
            min_sol_cost = cost
            min_solution = deepcopy(stations)
    print_solution(min_solution)
    return min_solution





def run_metaheristic(max_iterations=30000):
    curr_solution = find_initial_solution() # deepcopy(stations)
    curr_cost = solution_cost(curr_solution)
    history = [curr_cost]
    operator_names = ['MOVE', 'SWAP'] # '3-SWAP']
    
    for iteration in range(max_iterations):
        # if iteration == K//2: 
        #     weights=(1,0,10)
        #     curr_cost = solution_cost(new_solution, weights=weights)
        new_solution = deepcopy(curr_solution)
        i = rand(0, num_stations)
        j = rand(0, num_stations)
        operation = rand(0,len(operator_names))
        if operation == 0:
            new_solution = move_shop(new_solution, i, j)
        elif operation == 1:
            new_solution = swap_shops(new_solution, i, j)       
        else: 
            k = rand(0, num_stations)
            new_solution = three_swap_shops(new_solution, i, j, k)       
        
        new_cost = solution_cost(new_solution, weights=weights)
        if new_cost < curr_cost:
            print(f'it {iteration}. Improved cost [{operator_names[operation]}]: {new_cost}')
            solution_cost(new_solution, weights=weights, verbose=True)
            assert check_is_valid_solution(new_solution)
            curr_solution = new_solution
            curr_cost = new_cost
        history.append(curr_cost)
    best_solution = curr_solution
    
    global last_optimization_history
    last_optimization_history = history
    return best_solution


def optimize_logistics(max_iterations=30000):
    best_solution = run_metaheristic(max_iterations=max_iterations)
    print_solution(best_solution)
    package_dispersion = get_solution_dispersion(best_solution)//2
    print(f'Dispersión = {package_dispersion}')
    print(solution_cost(best_solution, weights=weights))
    return best_solution