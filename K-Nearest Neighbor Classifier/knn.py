import numpy as np


def load_data_into_map(path):
    return list(
        map(lambda s: s.strip().split(", "), open(path).readlines()))  # Get comma separated rows from training input


age_normalizer = 50  # Interpreted from training data


def form_mappings_from_training(data_map):
    mappings = {}
    for row in data_map:
        for index, val in enumerate(row):
            feature = (index, val)  # Feature e.g., (1, 'Self-emp-not-inc')
            if index not in [0, 7, 9]:  # Parse only categorical fields, skip age, hours-per-week and target
                if feature not in mappings:
                    mappings[feature] = len(mappings)  # (1, 'Self-emp-not-inc'): Serial feature #
    return mappings


def binarize_normalize_data(data_map, mappings):
    partial_binarized = []
    given_targets = []
    total_rows = 0
    for row in data_map:
        person_row = []
        for index, val in enumerate(row):
            if index not in [0, 7, 9]:
                feature = (index, val)
                if feature in mappings:
                    person_row.append(mappings[feature])
                else:
                    mappings[feature] = len(mappings)
            elif index in [0, 7]:
                person_row.append(float(val) / age_normalizer)
            elif index == 9:
                given_targets.append(val)
        partial_binarized.append(person_row)
        total_rows += 1

    binarized_vector = np.zeros((total_rows, len(mappings) + 2))
    for index, person_row in enumerate(partial_binarized):
        for i, val in enumerate(person_row):
            if i not in [0, 7]:
                binarized_vector[index][val] = 1
            elif i == 0:
                binarized_vector[index][-2] = val
            elif i == 7:
                binarized_vector[index][-1] = val
    return binarized_vector, given_targets, total_rows


def binarize_all_data(data_map, mappings):
    partial_binarized = []
    given_targets = []
    total_rows = 0
    for row in data_map:
        person_row = []
        for index, val in enumerate(row):
            if index != 9:
                feature = (index, val)
                if feature in mappings:
                    person_row.append(mappings[feature])
                else:
                    mappings[feature] = len(mappings)
            else:
                given_targets.append(val)
        partial_binarized.append(person_row)
        total_rows += 1

    binarized_vector = np.zeros((total_rows, len(mappings)))
    for index, person_row in enumerate(partial_binarized):
        for i, val in enumerate(person_row):
            binarized_vector[index][val] = 1
    return binarized_vector, given_targets, total_rows


def find_topk(k, manh_or_eucl, vector_train, vector_dev):
    dist = None
    diff = vector_train - vector_dev[:, np.newaxis]
    if manh_or_eucl == "eucl":
        dist = np.linalg.norm(diff, axis=2)
    elif manh_or_eucl == "manh":
        dist = np.sum(np.abs(diff), axis=2)
    train_rows = vector_train.shape[0]
    kVals = range(k) if k < train_rows else range(train_rows)
    top_k_indices = np.argpartition(dist, kVals, axis=1)[:, :k]
    top_k_dist = dist[np.arange(dist.shape[0])[:, None], top_k_indices]
    return top_k_indices, top_k_dist


def knn(k, manh_or_eucl, vector_train, vector_dev, train_targets):
    top_k_indices, top_k_dist = find_topk(k, manh_or_eucl, vector_train, vector_dev)
    predicted_list = []
    for row in top_k_indices:
        nearby_targets = []
        for index in row:
            nearby_targets.append(train_targets[index])
        predicted_list.append(max(set(nearby_targets), key=nearby_targets.count))
    return predicted_list


if __name__ == "__main__":
    data_map_train = load_data_into_map("hw1-data/income.train.txt.5k")
    mappings = form_mappings_from_training(data_map_train)
    vector_train, train_targets, train_rows = binarize_normalize_data(data_map_train, mappings)

    data_map_dev = load_data_into_map("hw1-data/income.dev.txt")
    vector_dev, dev_targets, dev_rows = binarize_normalize_data(data_map_dev, mappings)

    if vector_train.shape[1] != vector_dev.shape[1]:
        for i in range(vector_dev.shape[1] - vector_train.shape[1]):
            vector_train = np.insert(vector_train, i - 2, values=0, axis=1)

    '''data_map_test = load_data_into_map("hw1-data/income.test.blind")
    vector_test, test_targets, test_rows = binarize_normalize_data(data_map_test, mappings)

    if vector_train.shape[1] != vector_test.shape[1]:
        for i in range(vector_test.shape[1] - vector_train.shape[1]):
            vector_train = np.insert(vector_train, i - 2, values=0, axis=1)

    predicted_list = knn(41, "eucl", vector_train, vector_test, train_targets)

    test_positivity = 0

    with open('hw1-data/income.test.blind', 'r') as istr:
        with open('hw1-data/income.test.predicted', 'w') as ostr:
            for i, line in enumerate(istr):
                line = line.rstrip('\n')
                line = line + ', ' + predicted_list[i]
                print(line, file=ostr)
                if predicted_list[i] == '>50K':
                    test_positivity += 1
    print(test_positivity / test_rows * 100)'''

    ks = [1, 3, 5, 7, 9, 99, 999, 9999]
    #ks = [5]
    for k in ks:
        predicted_list = knn(k, "eucl", vector_train, vector_dev, train_targets)
        dev_positivity = 0
        dev_errors = 0
        for idx, val in enumerate(predicted_list):
            if val == ">50K":
                dev_positivity += 1
            if val != dev_targets[idx]:
                dev_errors += 1
        dev_positivity = dev_positivity / dev_rows * 100
        dev_errors = dev_errors / dev_rows * 100

        predicted_list_2 = knn(k, "eucl", vector_train, vector_train, train_targets)
        train_positivity = 0
        train_errors = 0
        for idx, val in enumerate(predicted_list_2):
            if val == ">50K":
                train_positivity += 1
            if val != train_targets[idx]:
                train_errors += 1
        train_positivity = train_positivity / train_rows * 100
        train_errors = train_errors / train_rows * 100
        #print("k = %d\t train_err %.2f%% (+:%.2f%%)" % (k, train_errors, train_positivity))
        print("k = %d\t train_err %.2f%% (+:%.2f%%)\tdev error %.2f%% (+:%.2f%%)" % (
            k, train_errors, train_positivity, dev_errors, dev_positivity))
