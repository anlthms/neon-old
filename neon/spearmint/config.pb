language: PYTHON
name:     "run"

# step size .001 or .002
# variable {
#  name: "steps"
#  type: FLOAT
#  size: 2
#  min:  1 # 1 # 0.5
#  max:  4 # 4 # 5
# }

# momentum
# variable {
#  name: "momenta"
#  type: FLOAT
#  size: 1
#  min:  0.5
#  max:  0.99
# }

# epochs
variable {
 name: "epochs"
 type: INT
 size: 1
 min:  20 # must be gib enough to reach a safe point I guess
 max:  200 # 135
}

#dicretness
variable {
 name: "q_weights" # going from -5:3 to -5:-2
 type: INT
 size: 5
 min:  -6
 max:  0
}

variable {
 name: "q_acts"  # going from 6:14 to 4:10
 type: INT
 size: 5
 min:  8
 max:  13
}

# variable {
#  name: "nonlins"
#  type: ENUM
#  size: 2
#  options:  "relu"
#  options:  "logistic"
# }
