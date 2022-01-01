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
import copy
import time
import numpy as np
from scipy import spatial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellular_automaton2 import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule


ALIVE = [1.0]
DEAD = [0]

STARTING_WIDTH = 10
STARTING_HEIGHT = 10

CANVAS_WIDTH = 60
CANVAS_HEIGHT = 60

class ConwaysCA(CellularAutomaton):
    """ Cellular automaton with the evolution rules of conways game of life """
    def __init__(self, canvas_width=CANVAS_WIDTH, canvas_height=CANVAS_HEIGHT, starting_width=STARTING_WIDTH, starting_height=STARTING_HEIGHT, initial_state=None, parents=None):
        self.starting_width = starting_width
        self.starting_height = starting_height
        self.start_width = canvas_width / 2 - starting_width / 2
        self.start_height = canvas_height / 2 - starting_height / 2
        self.initial_state = initial_state
        self.generation = None
        super().__init__(dimension=[canvas_width, canvas_height],
                         neighborhood=MooreNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

        self.previous_steps = []
        self.max_size = 0
        self.max_length = 0
        self.reset_state = self.get_cells()

    def hard_reset(self):
        self.__init__(self._dimension[0], self._dimension[1], self.starting_width, self.starting_height, self.reset_state)

    def init_cell_state(self, index):  # pylint: disable=no-self-use
        # If we have a predefined state then return it
        if self.initial_state:
            return self.initial_state[index].state

        if index[0] < self.start_height or index[0] > (self.start_height + self.starting_height):
            return [0.0]

        if index[1] < self.start_width or index[1] > (self.start_width + self.starting_width):
            return [0.0]

        init = random.randint(0, 3) == 0
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
        super().evolve(times)

        cells_as_array = self.convert_to_array(self.get_cells())

        if cells_as_array in self.previous_steps:
            return True

        # Calculate largest modes
        current_size = self.cell_count(cells_as_array)
        self.max_size = current_size if current_size > self.max_size else self.max_size

        current_length = self.get_length()
        self.max_length = current_length if current_length > self.max_length else self.max_length
        self.previous_steps.append(cells_as_array)
        return False

    def get_length(self):
        try:
            cells = self.get_cells()
            pts = []
            for index in cells:
                if cells[index].state[0]:
                    pts.append(index)

            pts = np.array(pts)

            candidates = pts[spatial.ConvexHull(pts).vertices]
            # get distances between each pair of candidate points
            dist_mat = spatial.distance_matrix(candidates, candidates)

            # get indices of candidates that are furthest apart
            i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

            x, y = (candidates[i], candidates[j])
            return ((x[0] - y[0])**2 + (x[1] - y[1])**2)**0.5
        except:
            return 0

    def get_full_state(self):
        # Get cell states include starting size
        return self, (self.start_width, self.start_height, self.starting_width, self.starting_height)

    @staticmethod
    def __count_alive_neighbours(neighbours):
        return sum(map(sum, neighbours))

    @staticmethod
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

    @staticmethod
    def cell_count(game_cells):
        return sum(map(sum, game_cells))



class GolWindow(CAWindow):
    def print_process_info(self, evolve_duration, draw_duration, evolution_step):
        self._draw_engine.fill_surface_with_color(((0, 0), (self._rect.width, 30)))
        self._draw_engine.write_text((10, 5), "CA: " + "{0:.4f}".format(evolve_duration) + "s")
        self._draw_engine.write_text((210, 5), "Display: " + "{0:.4f}".format(draw_duration) + "s")
        self._draw_engine.write_text((430, 5), "Step: " + str(evolution_step))
        self._draw_engine.write_text((660, 5), "Generation:" + str(self.generation))

if __name__ == "__main__":
    CAWindow(cellular_automaton=ConwaysCA(),
             window_size=(1000, 830)).run(evolutions_per_second=40)