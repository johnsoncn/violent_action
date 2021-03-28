import json

dataset='/home/administrator/Z/Datasets/External Datasets/TAO/TAO_DIR/annotations/train.json'
js=json.load(open(dataset))

cat_l=[]
cat_id_l=[]
for c in js['categories']:
	cat_l.append(c['name'])
	cat_id_l.append(int(c['id']))


first_id=cat_id_l[0]
cls2id={}
id2cls={}
for i,v in enumerate(cat_id_l):
	cat_id_l[i]=cat_id_l[i]-first_id #zero padding id: #WARNING if is integer and sequential
	cls2id[cat_l[i]]=cat_id_l[i]
	id2cls[cat_id_l[i]]=cat_l[i]
	
print(id2cls)
with open('/home/administrator/Desktop/cls2id.txt', 'w') as f:
	json.dump(cls2id, f)

with open('/home/administrator/Desktop/id2cls.txt', 'w') as f:
	json.dump(id2cls, f)

print("generated!")
	
