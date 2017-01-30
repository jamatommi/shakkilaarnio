#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
import re, sys, time
from itertools import count
from collections import OrderedDict, namedtuple

import random, math

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e8

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = 60000 - 8*2700
MATE_UPPER = 60000 + 8*2700

#for transposition table node types
EXACT = 0
ALPHA = 1
BETA = 2

QS_LIMIT = 150
EVAL_ROUGHNESS = 20

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' rnbqkbnr\n'  #  20 - 29
    ' pppppppp\n'  #  30 - 39
    ' ........\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ........\n'  #  70 - 79
    ' PPPPPPPP\n'  #  80 - 89
    ' RNBQKBNR\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)

'''initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' ..r...k.\n'  #  20 - 29
    ' ....R.p.\n'  #  30 - 39
    ' ...r....\n'  #  40 - 49
    ' ...p..Np\n'  #  50 - 59
    ' .......P\n'  #  60 - 69
    ' ..P.....\n'  #  70 - 79
    ' P.......\n'  #  80 - 89
    ' R......K\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)'''
'''initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' ..r...k.\n'  #  20 - 29
    ' ....Rppp\n'  #  30 - 39
    ' ...r....\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ...P....\n'  #  70 - 79
    ' ........\n'  #  80 - 89
    ' R......K\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)'''

###############################################################################
# M'''ove and evaluation tables
###############################################################################

N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, N+N, N+W, N+E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}

pst = {
    'P': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 198, 198, 198, 198, 198, 198, 198, 198, 0,
        0, 178, 198, 198, 198, 198, 198, 198, 178, 0,
        0, 178, 198, 198, 198, 198, 198, 198, 178, 0,
        0, 178, 198, 208, 218, 218, 208, 198, 178, 0,
        0, 178, 198, 218, 238, 238, 218, 198, 178, 0,
        0, 178, 198, 208, 218, 218, 208, 198, 178, 0,
        0, 178, 198, 198, 198, 198, 198, 198, 178, 0,
        0, 198, 198, 198, 198, 198, 198, 198, 198, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'B': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 797, 824, 817, 808, 808, 817, 824, 797, 0,
        0, 814, 841, 834, 825, 825, 834, 841, 814, 0,
        0, 818, 845, 838, 829, 829, 838, 845, 818, 0,
        0, 824, 851, 844, 835, 835, 844, 851, 824, 0,
        0, 827, 854, 847, 838, 838, 847, 854, 827, 0,
        0, 826, 853, 846, 837, 837, 846, 853, 826, 0,
        0, 817, 844, 837, 828, 828, 837, 844, 817, 0,
        0, 792, 819, 812, 803, 803, 812, 819, 792, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'N': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 627, 762, 786, 798, 798, 786, 762, 627, 0,
        0, 763, 798, 822, 83, 834, 822, 798, 763, 0,
        0, 817, 852, 876, 888, 888, 876, 852, 817, 0,
        0, 797, 832, 856, 868, 868, 856, 832, 797, 0,
        0, 799, 834, 858, 870, 870, 858, 834, 799, 0,
        0, 758, 793, 817, 829, 829, 817, 793, 758, 0,
        0, 739, 774, 798, 810, 810, 798, 774, 739, 0,
        0, 683, 718, 742, 754, 754, 742, 718, 683, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'R': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'Q': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'K': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 60098, 60132, 60073, 60025, 60025, 60073, 60132, 60098, 0,
        0, 60119, 60153, 60094, 60046, 60046, 60094, 60153, 60119, 0,
        0, 60146, 60180, 60121, 60073, 60073, 60121, 60180, 60146, 0,
        0, 60173, 60207, 60148, 60100, 60100, 60148, 60207, 60173, 0,
        0, 60196, 60230, 60171, 60123, 60123, 60171, 60230, 60196, 0,
        0, 60224, 60258, 60199, 60151, 60151, 60199, 60258, 60224, 0,
        0, 60287, 60321, 60262, 60214, 60214, 60262, 60321, 60287, 0,
        0, 60298, 60332, 60273, 60225, 60225, 60273, 60332, 60298, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}


###############################################################################
# Chess logic
###############################################################################

class Position(namedtuple('Position', 'board score wc bc ep kp')):
    """ A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    """

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if not p.isupper(): continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # Pawn move, double move and capture
                    if p == 'P' and d in (N, N+N) and q != '.': break
                    if p == 'P' and d == N+N and (i < A1+N or self.board[i+N] != '.'): break
                    if p == 'P' and d in (N+W, N+E) and q == '.' and j not in (self.ep, self.kp): break
                    # Move it
                    yield (i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in 'PNK' or q.islower(): break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j+E] == 'K' and self.wc[0]: yield (j+E, j+W)
                    if i == H1 and self.board[j+W] == 'K' and self.wc[1]: yield (j+W, j+E)

    def rotate(self):
        ''' Rotates the board, preserving enpassant '''
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119-self.ep if self.ep else 0,
            119-self.kp if self.kp else 0)

    def nullmove(self):
        ''' Like rotate, but clears ep and kp '''
        return Position(
            self.board[::-1].swapcase(), -self.score,
            self.bc, self.wc, 0, 0)

    def move(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i+1:]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, '.')
        # Castling rights, we move the rook or capture the opponent's
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == 'K':
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = put(board, A1 if j < i else H1, '.')
                board = put(board, kp, 'R')
        # Pawn promotion, double move and en passant capture
        if p == 'P':
            if A8 <= j <= H8:
                board = put(board, j, 'Q')
            if j - i == 2*N:
                ep = i + N
            if j - i in (N+W, N+E) and q == '.':
                board = put(board, j+S, '.')
        # We rotate the returned position, so it's ready for the next player
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][119-j]
        # Castling check detection
        if abs(j-self.kp) < 2:
            score += pst['K'][119-j]
        # Castling
        if p == 'K' and abs(i-j) == 2:
            score += pst['R'][(i+j)//2]
            score -= pst['R'][A1 if j < i else H1]
        # Special pawn stuff
        if p == 'P':
            if A8 <= j <= H8:
                score += pst['Q'][j] - pst['P'][j]
            if j == self.ep:
                score += pst['P'][119-(j+S)]
        return score

###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'depth flag score')

# The normal OrderedDict doesn't update the position of a key in the list,
# when the value is changed.
class LRUCache:
    '''Store items in the order the keys were last added'''
    def __init__(self, size):
        self.od = OrderedDict()
        self.size = size
        self._searched_nodes = 0

    def get(self, key, default=None):
        try: self.od.move_to_end(key)
        except KeyError: return default
        return self.od[key]

    def __setitem__(self, key, value):
        try: del self.od[key]
        except KeyError:
            if len(self.od) == self.size:
                self.od.popitem(last=False)
        self.od[key] = value

class Searcher:
    def __init__(self):
        self.tp_score = LRUCache(TABLE_SIZE)
        self.tp_move = LRUCache(TABLE_SIZE)
        #self.ttable = LRUCache(TABLE_SIZE)
        self.nodes = 0
        self.threshold = 0.001
        self._control = 3
        self._starttime = 0
    def search_negamax(self, pos, min_depth=0):
        #assuming playing black
        best_val = float('-inf')
        best_move = None
        prev_best = None
        moves = sorted(pos.gen_moves(), key=pos.value, reverse=True)

        self._searched_nodes = 0
        self._starttime = time.time()

        control = self._control
        
        best_moves = []

        for d in range(1, 1000):
            best_val = float('-inf')
            best_move = None 

            if time.time() - self._starttime > self._control and d >= min_depth:
                print ("searched nodes: " + str(self._searched_nodes) + " depth: " + str(d))
                print(str(time.time() - self._starttime)) 

                if len(best_moves):
                    final = best_moves.pop() 
                    return best_val, final
                else:
                    return best_val, best_move

            for move in moves:
                value = -self._search_negamax(pos.move(move), d, float('-inf'), float('inf'), 1)

                if value > best_val:
                    best_val = value
                    best_move = move
            if not best_move in best_moves:
                best_moves.append(best_move)    
            #print(d)
        return best_val, best_moves.pop()
    def search(self, pos, secs):
        self._control = secs
        score, move = self.search_negamax(pos)
        return move,score


    def _search_negamax(self, pos, depth, alpha, beta, color, q_search = False):
        self._searched_nodes += 1

        if pos.score <=-MATE_LOWER:
            return -MATE_UPPER
        if (depth < 1 and not q_search) or time.time() - self._starttime > self._control*10:
                return pos.score

        ttEntry = self.tp_score.get((pos, depth))
        if ttEntry:
            if ttEntry.flag == EXACT:
                return ttEntry.score
            elif ttEntry.flag == ALPHA:
                alpha = max(alpha, ttEntry.score)
            elif ttEntry.flag == BETA:
                beta = min(beta, ttEntry.score)

            if alpha >= beta:
                return ttEntry.score
        
        orig_alpha = alpha
        
        #moves = sorted(pos.gen_moves(), key=pos.value, reverse=True)

        qs = False

        def moves():
            #nullmove
            if any(c in pos.board for c in 'RBNQ'):
                yield None, -self._search_negamax(pos.nullmove(), depth-3, -beta, -alpha, -color, False)
            #killer
            killer = self.tp_move.get(pos)
            if killer:
                yield killer, -self._search_negamax(pos.move(killer), depth-1, -beta, -alpha, -color, False)
                
            for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
                if pos.value(move) > QS_LIMIT:
                    qs = True
                else:
                    qs = False                
                yield move, -self._search_negamax(pos.move(move), depth-1, -beta, -alpha, -color, qs)
        
        best_val = float('-inf')
        best_move = None
        for move, score in moves():
            if score > best_val:
                best_val = score
                best_move = move

            alpha = max(alpha, score)
            if alpha > beta:
                break            
        #print(str(depth) + "/" + str(best_move) + "/" + str(move))


        e_score = best_val
        if best_val <= orig_alpha:
            e_flag = BETA
        elif best_val >= beta:
            e_flag = ALPHA
        else:
            e_flag = EXACT

        e_depth = depth

        self.tp_score[(pos, depth)] = Entry(e_depth, e_flag, e_score)
        self.tp_move[pos] = best_move

        return best_val

###############################################################################
# User interface
###############################################################################

# Python 2 compatability
if sys.version_info[0] == 2:
    input = raw_input
    class NewOrderedDict(OrderedDict):
        def move_to_end(self, key):
            value = self.pop(key)
            self[key] = value
    OrderedDict = NewOrderedDict


def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)


def print_pos(pos):
    print()
    uni_pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
                  'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙', '.':'·'}
    for i, row in enumerate(pos.board.split()):
        print(' ', 8-i, ' '.join(uni_pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n\n')


def main():
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    searcher = Searcher()
    while True:
        print_pos(pos)

        if pos.score <= -MATE_LOWER:
            print("You lost")
            break

        # We query the user until she enters a (pseudo) legal move.
        move = None
        while move not in pos.gen_moves():
            match = re.match('([a-h][1-8])'*2, input('Your move: '))
            if match:
                move = parse(match.group(1)), parse(match.group(2))
            else:
                # Inform the user when invalid input (e.g. "help") is entered
                print("Please enter a move like g8f6")
        pos = pos.move(move)

        # After our move we rotate the board and print it again.
        # This allows us to see the effect of our move.
        print_pos(pos.rotate())

        if pos.score <= -MATE_LOWER:
            print("You won")
            break

        # Fire up the engine to look for a move.
        score, move= searcher.search_negamax(pos)

        if score == MATE_UPPER:
            print("Checkmate!")

        # The black player moves from a rotated position, so we have to
        # 'back rotate' the move before printing it.
        print("My move:", render(119-move[0]) + render(119-move[1]))
        pos = pos.move(move)


if __name__ == '__main__':
    main()

