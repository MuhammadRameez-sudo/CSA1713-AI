"""Lab 7: Genetic algorithm maximizing f(x) = x * sin(x) on [0, 20]."""
import math
import random

def fitness(x):
    return x * math.sin(x)

def main():
    random.seed(11)
    pop_size = 30
    generations = 25
    lo, hi = 0.0, 20.0
    population = [random.uniform(lo, hi) for _ in range(pop_size)]

    print("=== Program 07: Genetic Algorithm ===")
    print("Maximize f(x) = x * sin(x)   for x in [0, 20]")
    print(f"Population: {pop_size}    Generations: {generations}")
    print()

    best_x, best_f = None, -1e9
    for gen in range(1, generations + 1):
        scored = sorted(((fitness(x), x) for x in population), reverse=True)
        if scored[0][0] > best_f:
            best_f, best_x = scored[0]
        if gen in (1, 5, 10, 15, 25):
            print(f"  Gen {gen:2d}   best f = {scored[0][0]:7.3f}   at x = {scored[0][1]:6.3f}")
        elites = [x for _, x in scored[:6]]
        children = list(elites)
        while len(children) < pop_size:
            a, b = random.sample(elites, 2)
            child = (a + b) / 2
            if random.random() < 0.3:
                child += random.uniform(-1.0, 1.0)
            children.append(min(hi, max(lo, child)))
        population = children

    print()
    print(f"Best solution: x = {best_x:.4f}")
    print(f"Best fitness:  f(x) = {best_f:.4f}")
    print("Program completed successfully.")

if __name__ == "__main__":
    main()
