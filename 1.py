x = 200
y = 240

n = 0
for i in range(1, y):
    for j in range(1, y):
        for k in range(3, y, 3):
            if i+k+j == y and i*5 + j*3 + k//3 == x:
                n = n+1
                print(i, j, k)
if n==0:
    print(0)