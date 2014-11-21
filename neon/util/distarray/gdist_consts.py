# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
'''
Constants

'''

# Border codes: where is the local array relative to the global array
NORTH = 0  # e.g. local array is along the northern edge of the global array
EAST = 1
SOUTH = 2
WEST = 3

NORTHEAST = 4
SOUTHEAST = 5
SOUTHWEST = 6
NORTHWEST = 7

CENTER = 8
SINGLE = -1

# to send in reverse directions
send_dict = {0: 2,
             1: 3,
             2: 0,
             3: 1,
             4: 6,
             5: 7,
             6: 4,
             7: 5,
             }

# halos corresponding to each borderId
halo_dict = {0: [3, 1, 6, 2, 5],
             1: [0, 7, 3, 6, 2],
             2: [3, 7, 0, 4, 1],
             3: [0, 4, 1, 5, 2],
             4: [3, 6, 2],
             5: [3, 7, 0],
             6: [0, 4, 1],
             7: [1, 5, 2],
             8: [0, 1, 2, 3, 4, 5, 6, 7],
             -1: [],
             }

# posOffsets in cart comm space for each direction
pos_offsets = {0: [-1, 0],
               1: [0, 1],
               2: [1, 0],
               3: [0, -1],
               4: [-1, 1],
               5: [1, 1],
               6: [1, -1],
               7: [-1, -1],
               8: [0, 0],
               -1: [],
               }

halo_dir_names = {0: 'N',
                  1: 'E',
                  2: 'S',
                  3: 'W',
                  4: 'NE',
                  5: 'SE',
                  6: 'SW',
                  7: 'NW',
                  }
