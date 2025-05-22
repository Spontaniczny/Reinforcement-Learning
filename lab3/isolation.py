from abc import abstractmethod
from enum import Enum
import itertools
import random
import time
from typing import Optional, Protocol
import math


class Colour(Enum):
    RED = 'R'
    BLUE = 'B'

    def flip(self) -> 'Colour':
        return Colour.RED if self == Colour.BLUE else Colour.BLUE


class Board:
    def __init__(self, width: int, height: int):
        self.width: int = width
        self.height: int = height
        self.positions: dict[tuple[int, int], str] = dict()
        self.red_position: Optional[tuple[int, int]] = None
        self.blue_position: Optional[tuple[int, int]] = None
        self._prepare_board()

    def _prepare_board(self):
        for i in range(self.width):
            for j in range(self.height):
                self.positions[(i, j)] = '.'

    def __str__(self):
        representation = '\\ ' + ' '.join([str(i + 1) for i in range(self.width)]) + '\n'
        for j in range(self.height):
            representation += (chr(ord('A') + j) + ' ' + ' '.join([self.positions[i, j] for i in range(self.width)]))
            if j < self.height - 1:
                representation += '\n'
        return representation

    def moves_for(self, current_player: Colour) -> list[tuple[int, int]]:
        result = []
        player_position = self._player_position(current_player)
        if player_position is None:
            for position in self.positions:
                if self.positions[position] == '.':
                    result.append(position)
        else:
            directions = list(itertools.product([-1, 0, 1], repeat=2))
            directions.remove((0, 0))
            for dx, dy in directions:
                px, py = player_position
                px, py = px + dx, py + dy
                while 0 <= px < self.width and 0 <= py < self.height:
                    potential_position = px, py
                    if self.positions[potential_position] == '.':
                        result.append(potential_position)
                        px, py = px + dx, py + dy
                    else:
                        break
        return result

    def apply_move(self, current_player: Colour, move: tuple[int, int]) -> None:
        player_position = self._player_position(current_player)
        if player_position is not None:
            self.positions[player_position] = '#'
        self.positions[move] = current_player.value
        self._update_player_position(current_player, move)

    def _player_position(self, current_player: Colour) -> tuple[int, int]:
        return self.red_position if current_player == Colour.RED else self.blue_position

    def _update_player_position(self, current_player: Colour, new_position: tuple[int, int]) -> None:
        if current_player == Colour.RED:
            self.red_position = new_position
        else:
            self.blue_position = new_position

    def to_state_str(self) -> str:
        positions_in_order = []
        for j in range(self.height):
            for i in range(self.width):
                positions_in_order.append(self.positions[(i, j)])
        return f"{self.width}_{self.height}_{''.join(positions_in_order)}"

    @staticmethod
    def from_state_str(state_str: str) -> 'Board':
        width, height, positions = state_str.split('_')
        width, height = int(width), int(height)
        board = Board(width, height)
        for j in range(height):
            for i in range(width):
                position = positions[j * width + i]
                board.positions[(i, j)] = position
                if position == Colour.RED.value:
                    board.red_position = (i, j)
                elif position == Colour.BLUE.value:
                    board.blue_position = (i, j)
        return board

    def duplicate(self) -> 'Board':
        return self.from_state_str(self.to_state_str())


class Player(Protocol):
    @abstractmethod
    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def register_opponent_action(self, action: tuple[int, int]) -> None:
        raise NotImplementedError


class Game:
    # zasady: (Jarvis, przetłumacz mi to na angielski)
    #  * jest dwóch graczy, czerwony i niebieski, czerwony porusza się pierwszy
    #  * każdy gracz ma dokładnie jeden pionek w swoim kolorze ('R' lub 'B')
    #  * plansza jest prostokątem, w swoim pierwszym ruchu każdy gracz może położyć pionek na jej dowolnym pustym polu
    #  * w kolejnych ruchach gracze naprzemiennie przesuwają swoje pionki
    #     * pionki poruszają się jak hetmany szachowe (dowolna liczba pól w poziomie, pionie, lub po skosie)
    #     * pole, z którego pionek startował jest usuwane z planszy ('.' zastępuje '#') i trwale zablokowane
    #     * zarówno pionek innego gracza jak i zablokowane pola uniemożliwiają dalszy ruch (nie da się ich przeskoczyć)
    #  * jeżeli gracz musi wykonać ruch pionkiem, a nie jest to możliwe (każdy z ośmiu kierunków zablokowany)...
    #  * ...to taki gracz przegrywa (a jego przeciwnik wygrywa ;])

    def __init__(self, red: Player, blue: Player, board: Board, current_player: Colour = Colour.RED):
        self.red: Player = red
        self.blue: Player = blue
        self.board: Board = board
        self.current_player: Colour = current_player
        self.finished: bool = False
        self.winner: Optional[Colour] = None

    def run(self, verbose=False):
        if verbose:
            print()
            print(self.board)

        while not self.finished:
            legal_moves = self.board.moves_for(self.current_player)
            if len(legal_moves) == 0:
                self.finished = True
                self.winner = Colour.BLUE if self.current_player == Colour.RED else Colour.RED
                break

            player = self.red if self.current_player == Colour.RED else self.blue
            opponent = self.red if self.current_player == Colour.BLUE else self.blue
            move = player.choose_action(self.board, self.current_player)
            opponent.register_opponent_action(move)
            self.board.apply_move(self.current_player, move)
            self.current_player = self.current_player.flip()

            if verbose:
                print()
                print(self.board)

        if verbose:
            print()
            print(f"WINNER: {self.winner.value}")


class RandomPlayer(Player):
    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        legal_moves = board.moves_for(current_player)
        return random.sample(legal_moves, 1)[0]

    def register_opponent_action(self, action: tuple[int, int]) -> None:
        pass


class MCTSNode:
    def __init__(self, board: Board, current_player: Colour, c_coefficient: float):
        self.parent: Optional[MCTSNode] = None
        self.leaf: bool = True
        self.terminal: bool = False
        self.times_chosen: int = 0
        self.value: float = 0.5
        self.children: dict[tuple[int, int], MCTSNode] = dict()
        self.board: Board = board
        self.current_player: Colour = current_player  # kto na tej planszy robi ruch
        self.c_coefficient: float = c_coefficient

    def select(self, final=False) -> tuple[int, int]:
        if final:
            most_picked_action = []
            counter = -1

            for key, child in self.children.items():
                if child.times_chosen > counter:
                    counter = child.times_chosen
                    most_picked_action = [key]
                elif child.times_chosen == counter:
                    most_picked_action.append(key)

            return random.choice(most_picked_action)


        rewards: dict[int, tuple[int, int]] = dict()
        total_children_moves = 0

        for key, child in self.children.items():
            if child.times_chosen == 0:
                return key

            total_children_moves += child.times_chosen

        for key, child in self.children.items():
            reward = child.value + child.c_coefficient * math.sqrt(math.log(total_children_moves) / child.times_chosen)
            rewards[reward] = key

        return self.get_random_best(rewards)


    def expand(self) -> None:
        if not self.terminal and self.leaf:
            legal_moves = self.board.moves_for(self.current_player)
            if len(legal_moves) > 0:
                self.leaf = False
                oponent = self.current_player.flip()
                for move in legal_moves:
                    child_board = self.board.duplicate()
                    child_board.apply_move(self.current_player, move)
                    child = MCTSNode(child_board, oponent, self.c_coefficient)
                    child.parent = self
                    self.children[move] = child
            else:
                self.terminal = True

    def simulate(self) -> Colour:
        if not self.terminal:
            game = Game(RandomPlayer() , RandomPlayer(), self.board, current_player=self.current_player)
            game.run(verbose=False)

            return game.winner
        else:
            return self.current_player.flip()


    def backpropagate(self, winner: Colour) -> None:
        self.times_chosen += 1
        reward = -1 if winner == self.current_player else 1
        self.value += (reward - self.value) / self.times_chosen
        # self.value += (self.value - reward) / self.times_chosen


        if self.parent is not None:
            self.parent.backpropagate(winner)


    @staticmethod
    def get_random_best(rewards: dict[int, tuple[int, int]]):
        max_key = max(rewards.keys())
        return random.choice([item for key, item in rewards.items() if key == max_key])


class MCTSPlayer(Player):
    def __init__(self, time_limit: float, c_coefficient: float):
        self.time_limit: float = time_limit
        self.root_node: Optional[MCTSNode] = None
        self.c_coefficient: float = c_coefficient

    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        if self.root_node is None:
            self.root_node = MCTSNode(board.duplicate(), current_player, self.c_coefficient)

        start_time = time.time()
        while True:
            self._mcts_iteration()

            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= self.time_limit:
                break

        action = self.root_node.select(final=True)
        self._step_down(action)
        return action

    def register_opponent_action(self, action: tuple[int, int]) -> None:
        if self.root_node is not None:
            self.root_node.expand()
            self._step_down(action)

    def _mcts_iteration(self):
        node = self.root_node
        while not node.leaf:
            action = node.select()
            node = node.children[action]
        node.expand()
        winner = node.simulate()
        node.backpropagate(winner)

    def _step_down(self, action: tuple[int, int]) -> None:
        new_root = self.root_node.children[action]
        new_root.parent = None
        self.root_node = new_root


def main() -> None:
    red_wins = 0
    blue_wins = 0
    time_limit = 0.5
    c = 0.5

    for i in range(100):
        board = Board(8, 8)
        # red_player = RandomPlayer()
        # blue_player = RandomPlayer()

        blue_player = MCTSPlayer(time_limit, c)
        red_player = MCTSPlayer(0.2, 3)
        game = Game(red_player, blue_player, board)
        game.run(verbose=False)

        if game.winner == Colour.RED:
            red_wins += 1
        else:
            blue_wins += 1

        print(f"After {i} games status: {red_wins=}, {blue_wins=}")

    print(red_wins, blue_wins)


if __name__ == '__main__':
    main()


# My little schema for testing. I have put it here for you, my little friend of UMISI in the future.
# You will have to edit a little (as always) ((you are smart, you will figure it out!))
# It isn't perfect, but I hope it's enough <3

# import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor

# def simulate_games_wrapper(args: Tuple[float, float, float, int, str]) -> Tuple[float, float, float]:
#     c_red, c_blue, time_limit, no_games, tested_player = args
#
#     red_wins = 0
#     blue_wins = 0
#
#     for _ in range(no_games):
#         board = Board(8, 8)
#         red_player = MCTSPlayer(time_limit, c_red)
#         blue_player = MCTSPlayer(time_limit, c_blue)
#         game = Game(red_player, blue_player, board)
#         game.run(verbose=False)
#
#         if game.winner == Colour.RED:
#             red_wins += 1
#         else:
#             blue_wins += 1
#
#     win_percentage = (red_wins if tested_player == "Red_vs_Blue" else blue_wins) / no_games * 100
#     return c_red, c_blue, win_percentage
#
#
# def main() -> None:
#     c_tab = np.round(np.arange(0.2, 2.1, 0.4), 2)
#     no_games = 100
#     tested_player = "Red_vs_Blue"
#     time_limit = 0.5
#
#     # Create a list of argument tuples
#     tasks = [(c_red, c_blue, time_limit, no_games, tested_player) for c_red in c_tab for c_blue in c_tab]
#
#     results = []
#     with ProcessPoolExecutor() as executor:
#         for result in executor.map(simulate_games_wrapper, tasks):
#             results.append(result)
#             print(f"{result[0]=}, {result[1]=}, {result[2]=:.2f}% wins")
#
#     # Plot results for each c_red
#     for c_red in c_tab:
#         red_win_percentages = [res[2] for res in results if res[0] == c_red]
#         c_blues = [res[1] for res in results if res[0] == c_red]
#
#         plt.figure(figsize=(10, 5))
#         plt.plot(c_blues, red_win_percentages, marker='o',
#                  label=f'{tested_player} wins out of {no_games} games, c benchmark')
#         plt.title(f'{tested_player} with c_red={c_red}, time={time_limit}')
#         plt.xlabel("blue c_coefficient")
#         plt.ylabel(f'red wins % for {no_games} games')
#         plt.legend()
#         plt.tight_layout()
#
#         plot_path = f"{tested_player}_c_red_{c_red}_games_{no_games}_8x8_time_{time_limit}.png"
#         plt.savefig(plot_path)
