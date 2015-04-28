import numpy as np

A = np.random.randint(0, 100, (3000, 3000))
B = np.random.randint(0, 100, (3000, 3000))

print("Generate complete!")

C = np.dot(A, B)

print(C)