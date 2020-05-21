

import numpy as np

import sys


def read_dictionary(dictionary_path):
    dictionary_file = open(dictionary_path, "r")

    dictionary = dict()

    for word in dictionary_file:
        dictionary[word.split(' ')[0]] = int(word.split(' ')[1])

    return dictionary


def read_file(filename):
    data = []

    data_file = open(filename, "r")

    for line in data_file:
        data.append(line)

    return data


def initialize_variables(data, dictionary):
    Y = np.empty(0)
    X = []
    for row in data:

        # Extracting y which is the first element
        Y = np.append(Y, int(row.split("\t")[0]))

        # Adding the bias term to X
        x = np.append(np.array(row.split("\t")[1:]), [len(dictionary)])

        x_int = []

        for index in x:
            x_int.append(int(index.split(":")[0]))

        X.append(x_int)

    # W is (m+1)*1
    W = np.append(np.zeros(len(dictionary) + 1, dtype=float), [1])

    return Y, X, W


def sigma(W, x):
    summation = 0.0

    for feature in x:
        # Multipling with W[feature+1] since W is shifted by 1 when we roll in the bias
        # into the vector
        summation = summation + W[feature]

    # print(summation)
    # Adding the bias term
    # summation = summation + W[len(W)-1]

    sigmoid = np.exp(summation) / (1 + np.exp(summation))

    return sigmoid


def train_model(X, W, Y, num_epochs, alpha):
    for epoch_count in range(0, num_epochs):

        for sample in range(0, len(X)):

            predicted_value = sigma(W, X[sample])

            loss = Y[sample] - predicted_value

            temp_weight = W.copy()

            for feature in X[sample]:
                temp_weight[feature] = W[feature] + (alpha * loss)

            W = temp_weight.copy()

    return W



def calculate_accuracy(w, x, y):

    all_predictions = []

    incorrect = 0

    for sample in range(len(x)):

        predicted_value = 1 if sigma(w, x[sample]) > 0.5 else 0

        all_predictions.append(predicted_value)

        if(predicted_value!=int(y[sample])):

            incorrect+=1

    return all_predictions, (incorrect/len(y))


def initialize_testing_data(data, dictionary):
    Y = np.empty(0)
    X = []

    for row in data:

        # Extracting y which is the first element
        Y = np.append(Y, int(row.split("\t")[0]))

        # Adding the bias term to X
        x = np.append(np.array(row.split("\t")[1:]), [len(dictionary)])

        x_int = []

        for index in x:
            x_int.append(int(index.split(":")[0]))

        X.append(x_int)

    return Y, X


def write_output_to_file(file_to_create, data_to_write):

    index = 0

    with open(file_to_create, 'w+') as filehandle:

        for label in data_to_write:

            filehandle.write(str(label))

            if(index != (len(data_to_write)-1)):

                filehandle.write('\n')

            index = index + 1





if __name__ == "__main__":


    file_train_input = sys.argv[1]

    file_validation_input = sys.argv[2]

    file_test_input = sys.argv[3]

    dictionary_file = sys.argv[4]

    training_label_filename = sys.argv[5]

    testing_label_filename = sys.argv[6]

    metrics_output_filename = sys.argv[7]

    num_epochs = int(sys.argv[8])

    data_train = read_file(file_train_input)

    data_test = read_file(file_test_input)

    dictionary = read_dictionary(dictionary_file)

    alpha = 0.1

    Y_train, X_train, W = initialize_variables(data_train, dictionary)

    W = train_model(X_train, W, Y_train, num_epochs, alpha)

    Y_test, X_test = initialize_testing_data(data_test, dictionary)

    testing_result_labels, testing_accuracy = calculate_accuracy(W, X_test, Y_test)

    training_result_labels, training_accuracy = calculate_accuracy(W, X_train, Y_train)


    # Writing outputs to file
    write_output_to_file(training_label_filename, training_result_labels)

    write_output_to_file(testing_label_filename, testing_result_labels)


    fileWrite = open(metrics_output_filename, "w+")

    fileWrite.write("error(train): " + str(training_accuracy) + '\n')

    fileWrite.write("error(test): " + str(testing_accuracy))

    fileWrite.close()

