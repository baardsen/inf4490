from typing import Dict, Tuple, List, Callable, Sequence, Set, Any, Iterable, cast
import argparse
import itertools
import math
import matplotlib.pyplot as plt
import random
import textwrap
import time
		
class Statistics:
	def __init__(self, test_method:Callable[[], Tuple[float, object]], runs: int) -> None:
		times:List[float] = []
		values:List[float] = []
		self.all_data:List[object] = []
		self.lowest_value = math.inf
		self.highest_value = -math.inf
		for _ in range(runs):
			start = time.time()
			value, data = test_method()
			self.all_data.append(data)
			times.append(time.time() - start)
			if value < self.lowest_value:
				self.lowest_value = value
				self.lowest_data = data
			if value > self.highest_value:
				self.highest_value = value
				self.highest_data = data
			values.append(value)
		
		self.mean = sum(values)/len(values)
		self.std = math.sqrt(sum([(d-self.mean)**2 for d in values])/len(values))
		
		self.total_time = sum(times)
		self.mean_time = self.total_time / len(times)
		
Distances = Dict[Tuple[str, str], float]
def load_file(name: str) -> Tuple[List[str], Distances]:
	cities:List[str] = []
	distances:Distances = {}
	with open(name, 'r') as file:
		cities = file.readline().rstrip('\n').split(';')
		for idx1, line in enumerate(file):
			for idx2, val in enumerate(line.split(';')):
				distances[(cities[idx1], cities[idx2])] = float(val)
	return cities, distances

def distance(tour:Sequence[str], distances:Distances) -> float:
	length:float = distances[(tour[-1], tour[0])]
	for tuple in zip(tour[:-1], tour[1:]):
		length += distances[tuple]
	return length
	
def brute_force(cities:List[str], distances:Distances) -> Tuple[float, Sequence[str]]:
	shortest_tour = None
	shortest_distance = math.inf
	for candidate in itertools.permutations(cities):
		dist = distance(candidate, distances)
		if dist < shortest_distance:
			shortest_distance = dist
			shortest_tour = candidate
	return shortest_distance, shortest_tour
	
def hill_climb(cities: List[str], distances: Distances, no_change_limit: int, shuffle_initial=True) -> Tuple[float, List[str]]:
	shortest_distance = math.inf
	shortest_tour = cities[:]
	if shuffle_initial:
		random.shuffle(shortest_tour)
	shortest_distance = distance(shortest_tour, distances)
	no_change = 0
	while no_change < no_change_limit:
		candidate = shortest_tour[:]
		idx1, idx2 = rand2(candidate)
		candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
		dist = distance(candidate, distances)
		if dist < shortest_distance:
			shortest_distance = dist
			shortest_tour = candidate
			no_change = 0
		else:
			no_change += 1
	return shortest_distance, shortest_tour
	
def genetic_algorithm(cities: List[str], distances: Distances, 
		population_size: int, max_generations: int, no_change_limit: int, 
		tour_optimizer: Callable[[List[str]], List[str]] = None) -> Tuple[float, List[float]]:
	population:List[Tuple[float, List[str]]] = []
	for _ in range(population_size):
		candidate = random.sample(cities, len(cities))
		dist = distance(candidate, distances)
		population.append((dist, candidate))
	
	shortest_distances:List[float] = []
	shortest_distance = math.inf
	
	no_change = 0
	while no_change < no_change_limit and len(shortest_distances) < max_generations:
		sorted_pop = sorted(population, key=lambda tuple: tuple[0])
		if len(shortest_distances) == 0:
			shortest_distances.append(sorted_pop[0][0])
		parents = sorted_pop[:population_size//5]
		random.shuffle(parents)
		while len(parents) > 1:
			mother = parents.pop()[1]
			father = parents.pop()[1]
			
			child1, child2 = pmx(mother, father)
			child1 = inversion_mutation(child1)
			child2 = inversion_mutation(child2)
			
			if tour_optimizer is not None:
				tmp1 = tour_optimizer(child1)
				tmp2 = tour_optimizer(child2)
				d1 = distance(tmp1, distances)
				d2 = distance(tmp2, distances)
				if False: #Lamarckian/Baldwinian
					child1 = tmp1
					child2 = tmp2
			else:
				d1 = distance(child1, distances)
				d2 = distance(child2, distances)
			
			population.append((d1, child1))
			population.append((d2, child2))
			
		sorted_pop = sorted(population, key=lambda tuple: tuple[0])
		population = sorted_pop[:population_size]
		
		shortest_distances.append(sorted_pop[0][0])
		if shortest_distances[-1] < shortest_distance:
			shortest_distance = shortest_distances[-1]
			no_change = 0
		else:
			no_change += 1
	
	return shortest_distance, shortest_distances
	
def pmx(mother: List[str], father: List[str]) -> Tuple[List[str], List[str]]:
	l, u = rand2(mother)
	return pmx_child(mother, father, l, u), pmx_child(father, mother, l, u)
	
def pmx_child(mother: List[str], father: List[str], l:int, u:int) -> List[str]:
	child: List[str] = [None] * len(mother)
	child[l:(u+1)] = mother[l:(u+1)]
	for idx in range(l, u+1):
		if father[idx] not in child:
			idx2 = idx
			while child[idx2] is not None:
				idx2 = father.index(mother[idx2])
			child[idx2] = father[idx]
	for idx, val in enumerate(father):
		if val not in child:
			child[idx] = val
	return child

def inversion_mutation(tour: List[str]) -> List[str]:
	l, u = rand2(tour)
	return tour[:l] + tour[l:(u+1)][::-1] + tour[(u+1):]
	
def plot_average_fitness(stats: Statistics) -> None:
	fitness_per_run:List[List[float]] = cast(List[List[float]], stats.all_data)
	max_gens: int = max(map(lambda list: len(list), fitness_per_run))
	fitness_per_gen: List[List[float]] = [[fitness_per_run[j][min(i, len(fitness_per_run[j])-1)] for j in range(len(fitness_per_run))] for i in range(max_gens)]
	avg_per_gen:List[float] = [sum(fitness)/len(fitness) for fitness in fitness_per_gen]
	
	fig, ax = plt.subplots()
	ax.plot(range(1, max_gens+1), avg_per_gen, label='Average fitness')
	ax.legend()
	ax.set_ylabel('Fitness')
	ax.set_xlabel('Generations')
	ax.grid(True)
	plt.show()
	
def rand2(arr: List[Any]) -> Tuple[int, int]:
	indices = set(range(len(arr)))
	[l, u] = random.sample(indices, 2)
	if l > u:
		l, u = u, l
	return l, u
	
def main(method: str, city_count: int, runs: int, no_change_limit: int, population_size: int, max_generations: int) -> None:
	cities, distances = load_file('./european_cities.csv')
	cities = cities[0:city_count]
	test_method: Callable[[], Tuple[float, object]] = {
		'brute-force': lambda: brute_force(cities, distances),
		'hill-climb': lambda: hill_climb(cities, distances, no_change_limit),
		'ga': lambda: genetic_algorithm(cities, distances, population_size, max_generations, no_change_limit),
		'hybrid': lambda: genetic_algorithm(cities, distances, population_size, max_generations, no_change_limit, lambda tour: hill_climb(tour, distances, 5, False)[1])
	}[method]
	
	stats = Statistics(test_method, runs)

	print(textwrap.dedent(f'''\
		total time: {stats.total_time}
		mean time: {stats.mean_time}
		mean distance: {stats.mean}
		shortest distance: {stats.lowest_value}
		longest distance: {stats.highest_value}
		std: {stats.std}
		best tour: {stats.lowest_data}
	'''))
	
	if method in ('ga', 'hybrid'):
		plot_average_fitness(stats)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Inf4490 Oblig1')
	parser.add_argument('-m', '--method', choices=['brute-force', 'hill-climb', 'ga', 'hybrid'], required=True)
	parser.add_argument('-c', '--cities', type=int, choices=range(1, 25), default=6)
	parser.add_argument('-p', '--population', type=int, default=100)
	parser.add_argument('-g', '--generations', type=int, default=1000)
	parser.add_argument('-r', '--runs', type=int, default=1)
	parser.add_argument('--no-change', type=int, default=100)
	args = parser.parse_args()
	main(args.method, args.cities, args.runs, args.no_change, args.population, args.generations)