# Ergebnis von np.where() im JSON-Format speichern und laden
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

# Ergebnis von np.where() im npz-Format speichern und laden
def btn_save(self):
    # (24.07.2024)
    print('Save the areas to disk')
    start = time.process_time()

    # Build a dictionary with the coordinates of the areas
    dict1 = {}
    for i, area_i in enumerate(self.areas[1:]):
        unc_value = area_i['unc_value']
        indices = np.where(self.uncertainty == unc_value)
        key = 'Area %d' % (i + 1)
        dict1[key] = indices

    split_time = time.process_time()

    # Save the dictionary in npz-format
    filename = self.parent / 'areas.npz'
    with open(filename, 'wb') as file:
        np.savez(file, **dict1)

    end = time.process_time()
    print('split time: %f s' % (split_time - start))
    print('runtime: %f s' % (end - start))

def btn_reload(self):
    # (24.07.2024)
    print('Reload the areas from disk')

    filename = self.parent / 'areas.npz'
    with np.load(filename) as npzfile:
        print(npzfile.files)
        dict1 = {key: tuple(npzfile[key]) for key in npzfile}
