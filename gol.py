#!/usr/bin/env python3
"""
Copyright 2019 Richard Feistenauer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=wrong-import-position
# pylint: disable=missing-function-docstring

import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellular_automaton2 import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule


ALIVE = [1.0]
DEAD = [0]

STARTING_WIDTH = 20
STARTING_HEIGHT = 20

CANVAS_WIDTH = 100
CANVAS_HEIGHT = 100

def convert_to_array(game_cells):
    """ Helper function to work with arrays instead of dictionary on parsing the table """
    arr = []
    last_row = -1
    for cell in game_cells:
        if cell[0] != last_row:
            last_row = cell[0]
            arr.append([])

        arr[-1].append(int(game_cells[cell].state[0]))

    return arr

class ConwaysCA(CellularAutomaton):
    """ Cellular automaton with the evolution rules of conways game of life """

    def __init__(self):
        self.start_width = CANVAS_WIDTH / 2 - STARTING_WIDTH / 2
        self.start_height = CANVAS_HEIGHT / 2 - STARTING_HEIGHT / 2
        super().__init__(dimension=[CANVAS_WIDTH, CANVAS_HEIGHT],
                         neighborhood=MooreNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))



    def init_cell_state(self, index):  # pylint: disable=no-self-use
        rand = random.randrange(0, 16, 1)
        init = max(.0, float(rand - 14))

        if index[0] < self.start_height or index[0] > (self.start_height + STARTING_HEIGHT):
            return [0.0]

        if index[1] < self.start_width or index[1] > (self.start_width + STARTING_WIDTH):
            return [0.0]

        return [init]

    def evolve_rule(self, last_cell_state, neighbors_last_states):
        new_cell_state = last_cell_state
        alive_neighbours = self.__count_alive_neighbours(neighbors_last_states)

        # Too much or two little neighbors die
        if alive_neighbours < 2 or alive_neighbours > 3:
            return DEAD

        # Three neighbors always be alive
        if alive_neighbours == 3:
            return ALIVE

        # Two neighbors keep our state
        return last_cell_state

    def evolve(self, times=1):
        """ Override the default evolve so I can process each step """
        #print(convert_to_array(self.get_cells()))
        super().evolve(times)


    @staticmethod
    def __count_alive_neighbours(neighbours):
        count = 0
        for n in neighbours:
            if n == ALIVE:
                count += 1
        return count


if __name__ == "__main__":
    CAWindow(cellular_automaton=ConwaysCA(),
             window_size=(1000, 830)).run(evolutions_per_second=40)
