from typing import Dict, Tuple, List, Callable, Sequence
import numpy as np
import matplotlib.pyplot as plt
import argparse

def f(x: float) -> float:
	return -x**4 + 2*x**3 + 2*x**2 - x
	
def df(x: float) -> float:
	return -4*x**3 + 6*x**2 + 4*x - 1
	
def plot() -> None:
	X = np.linspace(-2, 3, 1000)
	fig, ax = plt.subplots()
	
	ax.plot(X, [f(x) for x in X], label='f(x)')
	ax.plot(X, [df(x) for x in X], label='f\'(x)')
	ax.legend()
	plt.show()
	
def main(exercise: str) -> None:
	if (exercise == 'c'):
		plot()
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Inf4490 Exercises 1')
	parser.add_argument('-e', '--exercise', choices=['c'], required=True)
	args = parser.parse_args()
	main(args.exercise)