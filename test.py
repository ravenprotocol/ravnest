

data_dict_keys = [0,1,2,3,4]
data_dict = {}

param_addresses = {'a':0, 'b':2}

param_address_mapping = {}
for i, address_to_param in enumerate(param_addresses.items()):
    if i < len(param_addresses) - 1:
        keys = data_dict_keys[data_dict_keys.index(address_to_param[1]):data_dict_keys.index(param_addresses[list(param_addresses.keys())[i+1]])]
    else:
        keys = data_dict_keys[data_dict_keys.index(address_to_param[1]):]
    
    for param_name in keys:
        param_address_mapping[param_name] = address_to_param[0]

print(param_address_mapping)

add = param_address_mapping.get(1)
del param_address_mapping[1]
print(add)