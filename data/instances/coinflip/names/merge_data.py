import os
import json


def read(file_name):
    with open(file_name, "r") as fp:
        text = fp.read()
        return text

file_names = [x for x in os.listdir() if x.startswith('yob')]

names = []
for file_name in file_names:
    file_data = read(file_name)
    file_data = file_data.split("\n")
    file_data = [x.split(",")[0] for x in file_data if x and int(x.split(",")[2])>50]
    names = names + file_data

names = list(set(names))

print(len(names))
with open('ssa_names_data.json',"w") as fp:
    json.dump(names, fp)
