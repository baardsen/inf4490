from typing import Dict, Tuple, List, Callable, Sequence
import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_x() -> Sequence[float]:
	return np.linspace(-2, 3, 1000)

def f(x: float) -> float:
	return -x**4 + 2*x**3 + 2*x**2 - x
	
def df(x: float) -> float:
	return -4*x**3 + 6*x**2 + 4*x - 1
	
def plot() -> None:
	X = generate_x()
	fig, ax = plt.subplots()
	
	ax.plot(X, [f(x) for x in X], label='f(x)')
	ax.plot(X, [df(x) for x in X], label='f\'(x)')
	ax.legend()
	ax.grid(True)
	plt.show()
	
def gradient_ascent(step: float, x: float, precision: float) -> Tuple[Sequence[float], Sequence[float], bool]:
	count = 0
	xs = [x]
	ys = [f(x)]
	dx = step * df(x)
	while (abs(dx) > precision and count < 50):
		x = x + dx
		xs.append(x)
		ys.append(f(x))
		dx = step * df(x)
	return xs, ys
	
def plot_gradient_ascent(start, step) -> None:
	X = generate_x()
	fig, ax = plt.subplots()
	ax.plot(X, [f(x) for x in X])
	xs, ys = gradient_ascent(step, start, 1e-10)
	ax.scatter(xs, ys, marker='|', color='red')
	ax.scatter([xs[-1]], [ys[-1]], marker='X', color='red')
	ax.grid(True)
	plt.show()
	
	
def main(exercise: str, start: float, step:float) -> None:
	{
		'b': lambda: plot(),
		'c': lambda: plot_gradient_ascent(start, step)
	}[exercise]()
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Inf4490 Exercises 1')
	parser.add_argument('-e', '--exercise', choices=['b', 'c'], required=True)
	parser.add_argument('--start', type=float, default=0)
	parser.add_argument('--step', type=float, default=0.1)
	args = parser.parse_args()
	main(args.exercise, args.start, args.step)