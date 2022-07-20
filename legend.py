from plant_classes import plant_classes

f = open("legend.txt", "w")
for idx, x in enumerate(plant_classes):
    f.write(f'{idx} - {x}\n')
f.close()
