
import numpy as np


raw_data = np.load('/H_share/data/FairDomain/Testing/data_08001.npz', allow_pickle=True)
mask = raw_data['disc_cup_mask']

print()
