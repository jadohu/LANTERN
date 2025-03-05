import json
import csv

with open('captions_val2017.json', 'r') as f:
    captions_data = json.load(f)

captions_by_image_id = {}
for item in captions_data['annotations']:
    image_id = item['image_id']
    if image_id not in captions_by_image_id:
        captions_by_image_id[image_id] = item['caption']
    else:
        if len(captions_by_image_id[image_id]) < len(item['caption']):
            captions_by_image_id[image_id] = item['caption']
        else:
            continue

captions = list(captions_by_image_id.values())
print(len(captions))

with open('captions_val2017_longest.json', 'w') as f_out:
    json.dump(captions, f_out, indent=4)