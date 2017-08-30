from typing import Dict, Tuple, List, Callable, Sequence
import numpy as np
import matplotlib as plot
import itertools
import argparse
import math
import time
import random
		
class Statistics:
	def __init__(self, test_method: Callable[[], Tuple[float, object]], runs: int) -> None:
		times = [] # type: List[float]
		values = [] # type: List[float]
		self.lowest_value = math.inf
		self.highest_value = -math.inf
		for _ in range(runs):
			start = time.time()
			value, data = test_method()
			times.append(time.time() - start)
			if value < self.lowest_value:
				self.lowest_value = value
				self.lowest_data = data
			if value > self.highest_value:
				self.highest_value = value
				self.highest_data = data
			values.append(value)
		
		self.best = min(values)
		self.worst = max(values)
		self.mean = sum(values)/len(values)
		self.std = math.sqrt(sum([(d-self.mean)**2 for d in values])/len(values))
		
		self.total_time = sum(times)
		self.mean_time = self.total_time / len(times)
		
Distances = Dict[Tuple[str, str], float]
def load_file(name: str, city_count: int) -> Tuple[List[str], Distances]:
	cities = [] # type: List[str]
	distances = {} # type: Distances
	with open(name, 'r') as file:
		cities = file.readline().rstrip('\n').split(';')
		for idx1, line in enumerate(file):
			for idx2, val in enumerate(line.split(';')):
				distances[(cities[idx1], cities[idx2])] = float(val)
	cities = cities[0:city_count]
	return cities, distances

	
def distance(tour: Sequence[str], distances: Distances) -> float:
	length = 0 # type: float
	for tuple in zip(tour[:-1], tour[1:]):
		length += distances[tuple]
	return length
	
def brute_force(cities: List[str], distances: Distances) -> Tuple[float, Sequence[str]]:
	shortest_tour = None
	shortest_distance = math.inf
	for candidate in itertools.permutations(cities):
		dist = distance(candidate, distances)
		if dist < shortest_distance:
			shortest_distance = dist
			shortest_tour = candidate
	return shortest_distance, shortest_tour
	
def hill_climb(cities: List[str], distances: Distances, no_change_limit: int) -> Tuple[float, List[str]]:
	shortest_distance = math.inf
	shortest_tour = cities[:]
	random.shuffle(shortest_tour)
	shortest_distance = distance(shortest_tour, distances)
	no_change = 0
	indices = set(range(len(shortest_tour)))
	while no_change < no_change_limit:
		candidate = shortest_tour[:]
		[idx1, idx2] = random.sample(indices, 2)
		candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
		dist = distance(candidate, distances)
		if dist < shortest_distance:
			shortest_distance = dist
			shortest_tour = candidate
			no_change = 0
		else:
			no_change += 1
	return shortest_distance, shortest_tour
	
def main(method: str, city_count: int, runs: int, no_change_limit: int) -> None:
	cities, distances = load_file('./european_cities.csv', city_count)
	if method == 'brute-force':
		test_method = lambda: brute_force(cities, distances)
	elif method == 'hill-climb':
		test_method = lambda: hill_climb(cities, distances, no_change_limit)
	
	stats = Statistics(test_method, runs)
	
	print((
		'total time: {}\n'
		'mean time: {}\n'
		'mean distance: {}\n'
		'shortest distance: {}\n'
		'longest distance: {}\n'
		'std: {}\n'
		'best tour: {}'
	).format(stats.total_time, stats.mean_time, 
		stats.mean, stats.best, stats.worst, stats.std, stats.lowest_data))
	
	#print(
	#	f'total time: {stats.total_time}\n'
	#	f'mean time: {stats.mean_time}\n'
	#	f'mean distance: {stats.mean}\n'
	#	f'shortest distance: {stats.best}\n'
	#	f'longest distance: {stats.worst}\n'
	#	f'std: {stats.std}\n'
	#	f'best tour: {stats.lowest_data}'
	#)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Inf4490 Oblig1')
	parser.add_argument('-m', '--method', choices=['brute-force', 'hill-climb', 'ga'], required=True)
	parser.add_argument('-c', '--cities', type=int, choices=range(1, 25), default=6)
	parser.add_argument('-r', '--runs', type=int, default=1)
	parser.add_argument('--no-change', type=int, default=100)
	args = parser.parse_args()
	main(args.method, args.cities, args.runs, args.no_change)