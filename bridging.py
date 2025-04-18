import numpy as np
import json

from TitanQ import base_path

# # subtracting 1 to match python indices vs julia indices (only needed to be done once)
# for lattice_size in range(2, 25, 2):
#
#     # Open and read the JSON file
#     with open(f'{base_path}/bonds/bonds_julia/all_bonds_{lattice_size**2}.json', 'r') as file:
#         data = json.load(file)
#
#         for v_ind in range(len(data)):
#             for h_ind in range(len(data[0])):
#
#                 data[v_ind][h_ind] -= 1
#     # print(data)
#     json_data = json.dumps(data)
#     with open(f'{base_path}/bonds/bonds_python/all_bonds_{lattice_size**2}.json', 'w') as file:
#         file.write(json_data)
