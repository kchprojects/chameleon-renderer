import json

j = None
with open("/home/karelch/Diplomka/rendering/chameleon-renderer/resources/setups/chameleon/lights.json","r") as file:
    j = json.load(file)

for l in j["lights"]:
    l["position"]["x"] *= -1
    l["direction"]["x"] *= -1
    
with open("ll.json","w") as file:
    json.dump(j,file,indent=4)