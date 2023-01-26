"""
The following program solves the numberlink puzzle.

It currently tries to solve the puzzle by filling the entire grid.
It is not yet able to solve the puzzle by filling only a part of the grid.

Additional requirement: pip install z3-solver

Takes as input an array of strings, where each string represents a row of the grid.


"""


from z3 import *
import numpy as np

def array_to_string_array(array):
    """
    Converts a numpy array to an array of strings. Each string represents a row of the grid.
    0s are replaced with spaces.
    """
    return [''.join([str(int(x)) if x != 0 else ' ' for x in row]) for row in array]

# Test array to string array

test_array = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 2],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 2, 0, 0, 1]])

assert array_to_string_array(test_array) == [
    '     ',
    '  1 2',
    '     ',
    '     ',
    ' 2  1']

def solve(puzzle=None):
    """
    Solves the puzzle by filling the entire grid.

    puzzle: an array of 0s and numbers, where 0 represents an empty cell and a number represents a cell with a number.
    """
    if puzzle is None:
        puzzle=[
        "         ",
        "   12  3 ",
        "     3 4 ",
        "         ",
        "    5 4 6",
        "         ",
        "         ",
        "      7  ",
        "012075  6"]

    puzzle2 = [
    " 32   ",
    " 4  1 ",
    "2     ",
    " 3    ",
    "      ",
    "  1  4"
    ]
    width=len(puzzle[0])
    height=len(puzzle)
    # number for each cell:
    cells=[[Int('cell_r%d_c%d' % (r,c)) for c in range(width)] for r in range(height)]
    # connections between cells. L means the cell has connection with cell at left , etc:
    L=[[Bool('L_r%d_c%d' % (r,c)) for c in range(width)] for r in range(height)]
    R=[[Bool('R_r%d_c%d' % (r,c)) for c in range(width)] for r in range(height)]
    U=[[Bool('U_r%d_c%d' % (r,c)) for c in range(width)] for r in range(height)]
    D=[[Bool('D_r%d_c%d' % (r,c)) for c in range(width)] for r in range(height)]
    200
    s=Solver()
    # U for a cell must be equal to D of the cell above , etc:
    for r in range(height):
        for c in range(width):
            if r!=0:
                s.add(U[r][c]==D[r-1][c])
            if r!=height -1:
                s.add(D[r][c]==U[r+1][c])
            if c!=0:
                s.add(L[r][c]==R[r][c-1])
            if c!=width -1:
                s.add(R[r][c]==L[r][c+1])
    # yes, I know , we have 4 bools for each cell at this point , and we can half this number ,
    # but anyway , for the sake of simplicity , this could be better.
    for r in range(height):
        for c in range(width):
            t=puzzle[r][c]
            if t==' ':
                # puzzle has space , so degree=2, IOW, this cell must have 2 connections , no more , no less.
                # enumerate all possible L/R/U/D booleans. two of them must be True , others are False.
                # TODO: 
                t=[]
                t.append(And(L[r][c], R[r][c], Not(U[r][c]), Not(D[r][c])))
                t.append(And(L[r][c], Not(R[r][c]), U[r][c], Not(D[r][c])))
                t.append(And(L[r][c], Not(R[r][c]), Not(U[r][c]), D[r][c]))
                t.append(And(Not(L[r][c]), R[r][c], U[r][c], Not(D[r][c])))
                t.append(And(Not(L[r][c]), R[r][c], Not(U[r][c]), D[r][c]))
                t.append(And(Not(L[r][c]), Not(R[r][c]), U[r][c], D[r][c]))
                s.add(Or(*t))
            else:
                # puzzle has number , add it to cells[][] as a constraint:
                s.add(cells[r][c]==int(t))
                # cell has degree=1, IOW, this cell must have 1 connection , no more , no less
                # enumerate all possible ways:
                t=[]
                t.append(And(L[r][c], Not(R[r][c]), Not(U[r][c]), Not(D[r][c])))
                t.append(And(Not(L[r][c]), R[r][c], Not(U[r][c]), Not(D[r][c])))
                t.append(And(Not(L[r][c]), Not(R[r][c]), U[r][c], Not(D[r][c])))
                t.append(And(Not(L[r][c]), Not(R[r][c]), Not(U[r][c]), D[r][c]))
                s.add(Or(*t))
            # if L[][]==True , cell's number must be equal to the number of cell at left , etc:
            if c!=0:
                s.add(If(L[r][c], cells[r][c]==cells[r][c-1], True))
            if c!=width -1:
                s.add(If(R[r][c], cells[r][c]==cells[r][c+1], True))
            if r!=0:
                s.add(If(U[r][c], cells[r][c]==cells[r-1][c], True))
            if r!=height -1:
                s.add(If(D[r][c], cells[r][c]==cells[r+1][c], True))
            # L/R/U/D at borders sometimes must be always False:

    for r in range(height):
        s.add(L[r][0]==False)
        s.add(R[r][width -1]==False)
    for c in range(width):
        s.add(U[0][c]==False)
        s.add(D[height -1][c]==False)
    # print solution:
    print(s.check())
    s.model()

    m=s.model()

    #print(m)
    print("")
    for r in range(height):
        for c in range(width):
            print(m[cells[r][c]], end=" ")
        print("")

    print("")

if __name__ == "__main__":
    solve(array_to_string_array(test_array))