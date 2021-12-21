from gol import *


MAX_EVOLVES = 500
POPULATION_SIZE = 10

MUTATION_PROBABILITY = 1/2
SWITCH_VALUE_PROBABILITY = 1/20

class evolver():
    def __init__(self, gol_class, size=POPULATION_SIZE):
        self.gol_class = gol_class
        self.population = [self.gol_class().get_full_state() for x in range(size)]

    def mutate_population(self):
        for gol, initial_state in self.population:
            (start_width, start_height, starting_width, starting_height) = initial_state
            if random.random() < MUTATION_PROBABILITY:
                for index in gol:
                    # Skip not init states
                    if (index[0] < start_height or index[0] > (start_height + starting_height)) or \
                        (index[1] < start_width or index[1] > (start_width + starting_width)):
                        continue

                    if random.random() < SWITCH_VALUE_PROBABILITY:
                        gol[index].state = [0.0] if gol[index].state == [1] else [1]


    def single_run(self):
        for i in range(5):
            gol = self.gol_class(initial_state=self.population[0][0])
            CAWindow(gol).run(evolutions_per_second=40)
            self.mutate_population()

    @staticmethod
    def display(gol):
        CAWindow(gol).run(evolutions_per_draw=0)

ev = evolver(ConwaysCA)
ev.single_run()