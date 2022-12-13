import os
import json
import jsonlines


i = 0
names = {}
for filename in os.listdir('names'):
    names_file = os.path.join('names', filename)
    with open(names_file, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        name,gender,count = line.split(",")
        count = int(count)
        if name not in names.keys():
            if gender == 'M':
                names.update({name: {'M': count, 'F': 0}})
            else:    
                names.update({name: {'M': 0, 'F': count}})
        else:
            if gender == 'M':
                names[name]['M'] = names[name]['M'] + count
            else:                     
                names[name]['F'] = names[name]['F'] + count

names = {k: v for k, v in sorted(names.items(), key=lambda item: item[1]['M'] + item[1]['F'], reverse=True)}
names_list = [{k:v} for k,v in names.items()]

with jsonlines.open('name_frequencies.jsonl', mode='w') as writer:
    writer.write_all(names_list)
writer.close()


