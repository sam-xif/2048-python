from tkinter import Frame, Label, CENTER

from src.game import constants as c
from src.game.gamestate import GameStateImpl


class WeightVisualizerGrid(Frame):
    def __init__(self, poller):
        Frame.__init__(self)

        self.grid()
        self.master.title('Weights')

        self.game_state = GameStateImpl()

        self.poller = poller

        # self.gamelogic = gamelogic
        self.commands = {c.KEY_UP: self.game_state.up, c.KEY_DOWN: self.game_state.down,
                             c.KEY_LEFT: self.game_state.left, c.KEY_RIGHT: self.game_state.right,
                             c.KEY_UP_ALT: self.game_state.up, c.KEY_DOWN_ALT: self.game_state.down,
                             c.KEY_LEFT_ALT: self.game_state.left, c.KEY_RIGHT_ALT: self.game_state.right,
                             c.KEY_H: self.game_state.left, c.KEY_L: self.game_state.right,
                             c.KEY_K: self.game_state.up, c.KEY_J: self.game_state.down}

        self.weights = [0 for i in range(16)]
        self.grid_cells = []
        self.init_grid()
        self.update_grid_cells()

        self.mainloop()


    def mainloop(self):
        while True:
            data = self.poller()
            if data is None:
                break

            # data is a 16-length array of weights
            self.weights = data
            self.update()
            self.update_grid_cells()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,
                           width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                             width=c.SIZE / c.GRID_LEN,
                             height=c.SIZE / c.GRID_LEN)
                cell.grid(row=i, column=j, padx=c.GRID_PADDING,
                          pady=c.GRID_PADDING)
                t = Label(master=cell, text="",
                          bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=c.FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def colorfunc(self, value):
        max_w = max([abs(x) for x in self.weights])
        if value < 0:
            r = (-255 / max_w) * value
            return '#{:02x}0000'.format(int(r))
        elif value > 0:
            g = (255 / max_w) * value
            return '#00{:02x}00'.format(int(g))
        else:
            return '#000000'

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.weights[4 * i + j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(
                        text="", bg=self.colorfunc(new_number))
                else:
                    self.grid_cells[i][j].configure(text="", bg=self.colorfunc(new_number))
        self.update_idletasks()

    def check_win_lose(self):
        """
        Returns True if the game is ended (either in a win or lose state)
        :return:
        """
        if self.game_state.state() == 'win':
            # self.grid_cells[1][1].configure(
            #     text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            # self.grid_cells[1][2].configure(
            #     text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            return True
        if self.game_state.state() == 'lose':
            # self.grid_cells[1][1].configure(
            #     text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            # self.grid_cells[1][2].configure(
            #     text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            return True

        return False

    def key_down(self, event):
        key = repr(event.char)
        # if key == c.KEY_BACK and len(self.history_matrixs) > 1:
        #     self.game_state.matrix = self.history_matrixs.pop()
        #     self.update_grid_cells()
        #     print('back on step total step:', len(self.history_matrixs))
        if key in self.commands:
            self.commands[repr(event.char)]()
            self.check_win_lose()
