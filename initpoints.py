import random

weights = [random.expovariate(1) for _ in range(100)]
w = sum(weights)
weights = [x/w for x in weights] 

f = open("Points and weights","w")
for i in range(100):
    x1 = random.random()
    x2 = random.random()
    x3 = random.random()
    x4 = random.random()
    x5 = random.random()
    x6 = random.random()
    x7 = random.random()
    x8 = random.random()
    x9 = random.random()
    x10 = random.random()

    f.write("{},{},{},{},{},{},{},{},{},{};{}\n".format(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,weights[i]))

f.write("\n")

weights = [random.expovariate(1) for _ in range(100)]
w = sum(weights)
weights = [x/w for x in weights] 

for i in range(100):
    x1 = random.random()
    x2 = random.random()
    x3 = random.random()
    x4 = random.random()
    x5 = random.random()
    x6 = random.random()
    x7 = random.random()
    x8 = random.random()
    x9 = random.random()
    x10 = random.random()

    f.write("{},{},{},{},{},{},{},{},{},{};{}\n".format(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,weights[i]))

f.close()
