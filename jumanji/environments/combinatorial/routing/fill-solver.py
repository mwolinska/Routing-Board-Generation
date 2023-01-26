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
    [3, 3, 0, 0, 0],
    [0, 0, 0, 1, 2],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [2, 4, 4, 0, 1]])

print(array_to_string_array(test_array))

assert array_to_string_array(test_array) == [
    '33   ',
    '   12',
    '     ',
    '     ',
    '244 1']

def solve(puzzle=None, wires = None):
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
        "812875  6"]

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
    # I think this should hold for non-fill solver too?
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
            # Want to stop the puzzle from adding more wires than the number of wires specified
            # Note: this doesn't solve the problem entirely, as we can get small squares of wires 
            # numbered 1 which are not connected to the pins. Doesn't affect solving the puzzle though.
            s.add(And(cells[r][c]<= wires, cells[r][c] >= 0))

            if t==' ':
                # puzzle has space , so degree=2, IOW, this cell must have 2 connections , no more , no less.
                # enumerate all possible L/R/U/D booleans. two of them must be True , others are False.
                t=[]
                t.append(And(L[r][c], R[r][c], Not(U[r][c]), Not(D[r][c]), cells[r][c]!=0))
                t.append(And(L[r][c], Not(R[r][c]), U[r][c], Not(D[r][c]), cells[r][c]!=0))
                t.append(And(L[r][c], Not(R[r][c]), Not(U[r][c]), D[r][c], cells[r][c]!=0))
                t.append(And(Not(L[r][c]), R[r][c], U[r][c], Not(D[r][c]), cells[r][c]!=0))
                t.append(And(Not(L[r][c]), R[r][c], Not(U[r][c]), D[r][c], cells[r][c]!=0))
                t.append(And(Not(L[r][c]), Not(R[r][c]), U[r][c], D[r][c], cells[r][c]!=0))
                # I added this constraint to allow empty cells. Check this works!
                t.append(And(Not(L[r][c]), Not(R[r][c]), Not(U[r][c]), Not(D[r][c]), cells[r][c]==0))
                # Also ensure we have valid numbers:
                #clause0 = If(And(Not(L[r][c]), Not(R[r][c]), Not(U[r][c]), Not(D[r][c])), cells[r][c]==0, True)
                s.add(Or(*t))
                
            else:
                # puzzle has number, add it to cells[][] as a constraint:
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
            # This gives us the constraint that cells must be equal to their neighbors. Definitely
            # necesary for all solvers.
            if c!=0:
                s.add(If(L[r][c], cells[r][c]==cells[r][c-1], True))
            if c!=width -1:
                s.add(If(R[r][c], cells[r][c]==cells[r][c+1], True))
            if r!=0:
                s.add(If(U[r][c], cells[r][c]==cells[r-1][c], True))
            if r!=height -1:
                s.add(If(D[r][c], cells[r][c]==cells[r+1][c], True))
    
    # L/R/U/D at borders sometimes must be always False:
    # This should hold even for non-fill puzzles.
    for r in range(height):
        s.add(L[r][0]==False)
        s.add(R[r][width -1]==False)
    for c in range(width):
        s.add(U[0][c]==False)
        s.add(D[height -1][c]==False)
    # print solution:
    #print(s)
    check = s.check()
    print(str(check))
    if str(check) == "sat":
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
    puzzle=[" 1 5  3  6",
"         2",
" 5        ",
"          ",
"7         ",
"    3  2  ",
" 4  6     ",
"        8 ",
"   19 7   ",
"4      89 "]


    solve(puzzle, 9)