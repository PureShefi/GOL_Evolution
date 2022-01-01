import random
import time
import matplotlib.pyplot as plt
from threading import Thread

random.seed(0)

from gol import *
from numpy.random import choice

MAX_EVOLVES = 500
MAX_GENERATIONS = 50
POPULATION_SIZE = 10

MUTATION_PROBABILITY = 1/2
SWITCH_VALUE_PROBABILITY = 1/15

PURGE_PERCENT=0.4


class evolver():
    def __init__(self, gol_class, generations=MAX_GENERATIONS, size=POPULATION_SIZE):
        self.gol_class = gol_class
        self.population = [self.gol_class().get_full_state() for x in range(size)]
        self.generations = generations
        self.average_fitness = []

    def mutate(self):
        """ Randomly change the population """

        # Only mutate the bottom samples in the population
        for gol, initial_state in self.population[len(self.population)//2:]:
            (start_width, start_height, starting_width, starting_height) = initial_state
            if random.random() < MUTATION_PROBABILITY:
                for index in gol.reset_state:
                    # Skip not init states
                    if (index[0] < start_height or index[0] > (start_height + starting_height)) or \
                        (index[1] < start_width or index[1] > (start_width + starting_width)):
                        continue

                    if random.random() < SWITCH_VALUE_PROBABILITY:
                        gol.reset_state[index].state = [0.0] if gol.reset_state[index].state == [1] else [1]

    @staticmethod
    def calculate_fitness(game):
        """ Get the metushlach value of a game """
        return int(game[0].max_size * game[0].max_length + game[0]._evolution_step + 1)

    def purge(self, percent=PURGE_PERCENT):
        """ Remove the worst population """
        starting_size = len(self.population)
        self.population.sort(key=self.calculate_fitness, reverse=True)
        self.population = self.population[:int(starting_size * percent)]

        return starting_size - len(self.population)

    def get_random_parents(self, count=2):
        """ Select random samples from the population """
        total_fitness = sum(map(self.calculate_fitness, self.population))
        probability_distribution = [self.calculate_fitness(sample)/total_fitness for sample in self.population]

        a, b = choice(range(len(self.population)), count, p=probability_distribution)
        return self.population[a], self.population[b]

    def breed(self, count):
        """ Add children from parents based on their fitness """
        for i in range(count):
            parents = self.get_random_parents(2)
            c_a, c_b = list(parents[0][0].get_cells().items()), list(parents[1][0].get_cells().items())
            initial_state = {**dict(c_a[:len(c_a)//2]), **dict(c_b[len(c_b)//2:])}
            child = self.gol_class(initial_state=initial_state).get_full_state()
            self.population.append(child)

    def evolve(self):
        """ Implement the genetic evlolving of the population """
        removed = self.purge()
        self.breed(removed)
        self.mutate()

    def single_run(self):
        """ Calculate all the evolution and search for metushalchs """
        # Reset population to original status
        for sample in self.population:
            sample[0].hard_reset()

        # evolve each sample of the population
        total = len(self.population)
        for count, (sample, initial_state) in enumerate(self.population):
            print("\r[*] Calculating sample {}/{}".format(count, total), end="")
            metushalch = False
            for i in range(MAX_EVOLVES):
                if sample.evolve():
                    metushalch = True
                    break

            # If it was not a metushlach, hard reset so its fitness will go down
            if not metushalch:
                sample.hard_reset()

            print (" fitness: {}".format(self.calculate_fitness((sample, None))))

        print("")

    def get_average_fitness(self):
        total_fitness = sum(map(self.calculate_fitness, self.population))
        return total_fitness / len(self.population)

    def run(self):
        """ Run multiple generations of the population and print the best """

        Z = [[0 for x in range(60)] for x in range(60)]
        Z = np.array(Z)
        X, Y = np.meshgrid(np.arange(Z.shape[1])+.5, np.arange(Z.shape[0])+.5)
        fig, axes = plt.subplots(2, self.generations // 2, figsize=(15, 10))

        for i, ax in enumerate(axes.flat):
            print("[*] Starting generation {}".format(i))
            self.single_run()
            self.average_fitness.append(self.get_average_fitness())
            self.evolve()

            ax.scatter(X[Z > 0], Y[Z > 0], color="k")

            ax.grid(True, color="k")
            ax.xaxis.set_major_locator(mticker.MultipleLocator())
            ax.yaxis.set_major_locator(mticker.MultipleLocator())
            ax.tick_params(size=0, length=0, labelleft=False, labelbottom=False)
            ax.set(xlim=(0, Z.shape[1]), ylim=(Z.shape[0], 0),
                   title='Generation {}'.format(i+1), aspect="equal")

        plt.tight_layout()
        plt.show(block=False)
        self.show_statics()

    def show_generation(self, generation):
        self.population.sort(key=self.calculate_fitness, reverse=True)
        gol = self.population[0][0]
        print("Starting generation 0")
        Thread(target=self.display, args=(gol, generation)).start()
        print("Ending generation")

    def show_statics(self):
        plt.plot(range(self.generations), self.average_fitness)
        plt.title('Average fitness per generation')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()

        self.population.sort(key=self.calculate_fitness, reverse=True)
        self.display(self.population[0][0], len(self.generations))

    @staticmethod
    def display(gol, generation):
        window = GolWindow(gol)
        window.generation = generation
        window.run(evolutions_per_draw=0)


ev = evolver(ConwaysCA, generations=2)
ev.run()