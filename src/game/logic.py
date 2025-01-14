#
# CS1010FC --- Programming Methodology
#
# Mission N Solutions
#
# Note that written answers are commented out to allow us to run your
# code easily while grading your problem set.
import random
from src.game import constants as c


#######
#Task 1a#
#######

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 1 mark for creating the correct matrix


def new_game(n):
    matrix = []

    for i in range(n):
        matrix.append([0] * n)
    return matrix

###########
# Task 1b #
###########

# [Marking Scheme]
# Points to note:
# Must ensure that it is created on a zero entry
# 1 mark for creating the correct loop
import itertools

def add_two(mat):
    choices = list(itertools.chain.from_iterable([[(i, j) for j in range(len(mat[0])) if mat[i][j] == 0] for i in range(len(mat))]))
    x, y = random.choice(choices)
    mat[x][y] = 2
    return mat

###########
# Task 1c #
###########

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 0 marks for completely wrong solutions
# 1 mark for getting only one condition correct
# 2 marks for getting two of the three conditions
# 3 marks for correct checking


def game_state(mat):
    # Commenting this out because 1) we don't want to stop at 2048 and 2) optimization
    # for i in range(len(mat)):
    #     for j in range(len(mat[0])):
    #         if mat[i][j] == 2048:
    #             return 'win'

    # According to the profiler, the method in which 8.3% of time is being spent (the highest proportion) is this one.
    # Time to optimize this more!

    for i in range(len(mat)-1):
        # intentionally reduced to check the row on the right and below
        # more elegant to use exceptions but most likely this will be their solution
        for j in range(len(mat[0])-1):
            # check for zero here as well since we check in the next loop
            if mat[i][j] == 0:
                return 'not over'

            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'not over'
    j = len(mat[0]) - 1
    for i in range(len(mat)):  # check for any zero entries
        if mat[i][j] == 0:
            return 'not over'
    i = len(mat) - 1
    for j in range(len(mat[0])):
        if mat[i][j] == 0:
            return 'not over'
    for k in range(len(mat)-1):  # to check the left/right entries on the last row
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1):  # check up/down entries on last column
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'

###########
# Task 2a #
###########

# [Marking Scheme]
# Points to note:
# 0 marks for completely incorrect solutions
# 1 mark for solutions that show general understanding
# 2 marks for correct solutions that work for all sizes of matrices


def reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new

###########
# Task 2b #
###########

# [Marking Scheme]
# Points to note:
# 0 marks for completely incorrect solutions
# 1 mark for solutions that show general understanding
# 2 marks for correct solutions that work for all sizes of matrices


def transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new

##########
# Task 3 #
##########

# [Marking Scheme]
# Points to note:
# The way to do movement is compress -> merge -> compress again
# Basically if they can solve one side, and use transpose and reverse correctly they should
# be able to solve the entire thing just by flipping the matrix around
# No idea how to grade this one at the moment. I have it pegged to 8 (which gives you like,
# 2 per up/down/left/right?) But if you get one correct likely to get all correct so...
# Check the down one. Reverse/transpose if ordered wrongly will give you wrong result.


def cover_up(mat):
    new = []
    for j in range(c.GRID_LEN):
        partial_new = []
        for i in range(c.GRID_LEN):
            partial_new.append(0)
        new.append(partial_new)
    done = False
    for i in range(c.GRID_LEN):
        count = 0
        for j in range(c.GRID_LEN):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return (new, done)


def merge(mat):
    points = 0
    done = False
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN-1):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                points += mat[i][j] * 2
                mat[i][j] *= 2
                mat[i][j+1] = 0
                done = True
    return (mat, done, points)


def up(game):
    #print("up")
    # return matrix after shifting up
    game = transpose(game)
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = transpose(game)
    points = temp[2]
    return (game, done, points)


def down(game):
    #print("down")
    game = reverse(transpose(game))
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = transpose(reverse(game))
    points = temp[2]
    return (game, done, points)


def left(game):
    #print("left")
    # return matrix after shifting left
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    points = temp[2]
    return (game, done, points)


def right(game):
    #print("right")
    # return matrix after shifting right
    game = reverse(game)
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = reverse(game)
    points = temp[2]
    return (game, done, points)
