language: PYTHON
name:     "spear_wrapper"


# In this simple example, autotune one stepsize and two nin parameters.

variable {
 name: "stepsize"
 type: FLOAT
 size: 1
 min:  0.1
 max:  1.0
}


variable {
 name: "epochs"
 type: INT
 size: 1
 min:  100
 max:  200
}



variable {
 name: "numberneurons"  #
 type: INT
 size: 1
 min:  2
 max:  20
}

# variable {
#  name: "nonlins"
#  type: ENUM
#  size: 2
#  options:  "relu"
#  options:  "logistic"
# }
