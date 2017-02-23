import sys
import numpy as np

matrixA = np.loadtxt(sys.argv[1], delimiter=',')
matrixB = np.loadtxt(sys.argv[2], delimiter=',')

result = np.dot(matrixA, matrixB)
result = sorted(result.astype(np.uint32))

with open("ans_one.txt", "w") as f:
    for ans in result:
        f.write(str(ans)+"\n")
