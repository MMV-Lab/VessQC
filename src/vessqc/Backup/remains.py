def btn_save(self):
    # (10.07.2024)
    print('Save the areas to disk')
    start = time.process_time()

    # Build a dictionary with the coordinates of the areas
    dict1 = {}
    for i, value in enumerate(self.uncert_values[1:]):
        indices = np.where(self.uncertainty == value)
        indices2 = (indices[0].tolist(), indices[1].tolist(), \
            indices[2].tolist())
        key = 'Area ' + str(i+1)
        dict1[key] = indices2

    # Save the dictionary in JSON format
    filename = self.parent / 'areas.json'
    with open(filename, 'w') as file:
        json.dump(dict1, file)

    end = time.process_time()
    print('runtime:', end - start, 'sec.')

def btn_reload(self):
    # (10.07.2024)
    print('Reload the areas')

    filename = self.parent / 'areas.json'
    with open(filename, 'r') as file:
        dict1 = json.load(file)

    for key in dict1:
        indices = dict1[key]
        indices2 = tuple(indices)
        dict1[key] = indices2