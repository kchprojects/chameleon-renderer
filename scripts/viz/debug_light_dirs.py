from viz_3d import Arrow3D
from matplotlib import pyplot as plt
import json

directions = []
positions = []
with open("../../resources/setups/chameleon/lights.json","r") as file:
    j = json.load(file);
    for l_j in j["lights"]:
        positions.append([l_j["position"][i] for i in "xyz"])
        directions.append([l_j["direction"][i] for i in "xyz"])
        
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(positions)):
    z = Arrow3D(*[[positions[i][x], positions[i][x]+directions[i][x]] for x in range(3)], mutation_scale=20,
                lw=3, arrowstyle="-|>", color="b")
    ax.add_artist(z)

ax.set_xlim(-500,500)
ax.set_ylim(-500,500)
ax.set_zlim(-500,500)
plt.draw()
plt.show()
