A = [[0, 0, 1, 0],
     [0, 2, 0, 0],
     [4, 7, 0, 0],
     [0, 0, 0, 5]]

B = [[2, 3, 5, 1],
     [4, 1, 6, 9],
     [4, 5, 2, 8],
     [3, 6, 9, 8]]

C = [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]

hdr = [0, 1, 2, 4, 5]
idx = [2, 1, 0, 1, 3]
dat = [1, 2, 4, 7, 5]

N = len(hdr) - 1
i = 0
# the complexity of these two loops together is O(M)
# where M is the number of non zeros.
# Then this option is better only if M < N
for row in range(0, N):
    while i < hdr[row+1]:
        col = idx[i]
        for k in range(N):
            C[row][k] += dat[i] * B[col][k]
            print(f"C[{row}][{k}] += dat[{i}] * B[{col}][{k}] = {dat[i]} * {B[col][k]}")
        i += 1

print(C)
