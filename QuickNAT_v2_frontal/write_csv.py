import csv

a = [0.99575769,0.92806156, 0.86727156, 0.88572313, 0.76512821, 0.86674454,
 0.88786417, 0.8241048,  0.78849626, 0.85923609, 0.84120172, 0.79568442,
 0.71875,    0.88197697, 0.85375218, 0.83193015, 0.65101038, 0.74987107,
 0.80630117, 0.47863248, 0.91937262, 0.8728686,  0.85645064, 0.79906542,
 0.84640602, 0.8944045, 0.91269841, 0.88918416, 0.91487271, 0.80452489,
 0.86537411, 0.80763116, 0.80974125, 0.84103318, 0.63874346, 0.53660797,
 0.56384065, 0.5476025,  0.35855856, 0.6031614]
b = [str(x) for x in a]
# label_map = dict()
brain_name = []
with open('/data/brain_segmentation/label_index_map_new.csv') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.strip().split(',')
        brain_name.append(line[0])
        # label_map[int(line[2])] = int(line[1])

with open('/data/brain_segmentation/test_cadni.csv') as f, open('/home/xiaodong/dice.csv', 'w') as out:
    out.write(','.join(['name'] + brain_name))
    out.write('\n')
    for i, line in enumerate(f):
        if i == 0:
            # out.write(','.join(['name'] + brain_name))
            # out.write('\n')
            continue
        line = line.strip().split(',')
        out.write(','.join([line[0].split('/')[-1]] + b))
        out.write('\n')
