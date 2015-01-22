language: PYTHON 
name: "neon.hyperopt.spear_wrapper"

variable {
    name: "lr"
    type: FLOAT
    size: 1
    min:  0.1
    max:  0.01
    }

variable {
    name: "nep"
    type: INT
    size: 1
    min:  1000
    max:  100
    }

