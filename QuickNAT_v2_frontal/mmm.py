import numpy as np
import pandas as pd
import os
import operator
import statistics
a = ['0.9952066870918801', '0.8540557506556198', '0.7939041430675174', '0.8183773816715062', '0.5321100916210756', '0.8366792391578997', '0.8890171223149286', '0.8203753351157559', '0.8232842817968058', '0.8101171656982507', '0.7560917908503408', '0.6361940297913997', '0.845814977948728', '0.8617185093908735', '0.7553263672037546', '0.7569463177567589', '0.4955465586643282', '0.6490850376399846', '0.7702875771245745', '0.0', '0.8869420613853929', '0.8132803048574881', '0.7711670480402142', '0.5425950195881265', '0.8238864800913525', '0.8927079624491826', '0.8873967635742469', '0.8685685810281728', '0.8205174111465354', '0.7912926391204826', '0.7392756915576737', '0.7553934973942451', '0.7068403908411053', '0.8049971606950028', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
l = ['0.9952179315050026', '0.8524064012788191', '0.7864933092632092', '0.8150750558796699', '0.479999999872', '0.8476792888244821', '0.8962711365155274', '0.8327155917912483', '0.8352710646527072', '0.8087750682533774', '0.7426503844246348', '0.6259398495652312', '0.8452309537838864', '0.8647189494675231', '0.7632041064330244', '0.7355395683241571', '0.4907481898237531', '0.6654611211172115', '0.7706546275308052', '0.0', '0.8806942629944012', '0.8176380447575162', '0.7677852348846063', '0.5973254085293722', '0.828926199410796', '0.900560344088198', '0.8886673313334086', '0.8537074148189666', '0.8139636873887031', '0.7739207832493565', '0.7220179997373303', '0.745773732095391', '0.7020453288721921', '0.7936079225659057', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ventricle1 = ['Left-Lateral-Ventricle','Right-Lateral-Ventricle']
ventricle2 = ['3rd-Ventricle', '4th-Ventricle']
ganglia = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper', 'Left-Caudate', 'Right-Caudate', 'Left-Putamen', 'Right-Putamen', 'Left-Pallidum', 'Right-Pallidum', 'Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'CSF']
total = ventricle1+ventricle2+ganglia
hippo = ['Left-Hippocampus', 'Right-Hippocampus']
path = r'C:\Users\Administrator\Desktop\csv'
dice_3d = pd.read_csv(os.path.join(path, 'dice/3d_dice.csv'))
dice_deeplab = pd.read_csv(os.path.join(path, 'dice/deeplab_dice.csv'))

vol_3d = pd.read_csv(os.path.join(path, 'vol/3d_vol.csv'))
vol_cat = pd.read_csv(os.path.join(path, 'vol/cat_vol_n.csv'))
vol_deep = pd.read_csv(os.path.join(path, 'vol/deeplab_vol.csv'))
vol_man = pd.read_csv(os.path.join(path, 'vol/manual_vol.csv'))

# print(dice_3d['Unknown'])
# td_dict = dict()
# td_list = []
# for region in total:
#     td_dict[region] = dice_3d[region].mean()
#     td_list.append(dice_3d[region].mean())
# sorted_td = sorted(td_dict.items(), key=operator.itemgetter(1))
# print(sorted_td)
td_dict = dict()
td_list = []
for region in ganglia:
    td_dict[region] = (vol_cat[region].corr(vol_man[region]))
    td_list.append(vol_cat[region].corr(vol_man[region]))
sorted_td = sorted(td_dict.items(), key=operator.itemgetter(1))
print(sorted_td)
print(sum(td_list)/len(td_list))
print(statistics.stdev(td_list))



