from sys import argv


in_file = argv[1]
out_file = argv[2]

points = []
normals = []
with open(in_file,"r") as file:
    for line in file.readlines() :
        if(line[0]=="#"):
            continue
        spl = line.split(" ")
        points.append([spl[i] for i in range(1,4)])
        normals.append([spl[i] for i in range(1,4)])

with open(out_file,"w") as file:
    for p in points:
        file.write("v " + " ".join(p) + "\n")
