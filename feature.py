import numpy as np

import sys


def read_dictionary(dictionary_file_input):

    dictionary_file = open(dictionary_file_input, "r")

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


def feature_vector_method_1(data):
    # if method = 1
    feature_vectors = []

    for line in data:

        y = line.split('\t')[0]

        words = line.split('\t')[1].split(' ')

        feature_vector = dict()

        for word in words:

            if word in dictionary:
                feature_vector[dictionary[word]] = 1

        feature_vectors.append(feature_vector)

    return feature_vectors


def feature_vector_method_2(data, dictionary):
    # if method = 2
    untrimmed_feature_vectors = []

    for line in data:

        y = line.split('\t')[0]

        words = line.split('\t')[1].split(' ')

        feature_vector = dict()

        for word in words:

            if word in dictionary:

                if dictionary[word] in feature_vector:

                    feature_vector[dictionary[word]] = feature_vector[dictionary[word]] + 1
                else:

                    feature_vector[dictionary[word]] = 1

        untrimmed_feature_vectors.append(feature_vector)

    trimmed_feature_vectors = []

    threshold = 4

    for dictionary in untrimmed_feature_vectors:

        trimmed_dictionary = dict()

        for key, value in dictionary.items():

            if value < threshold:
                trimmed_dictionary[key] = 1

        trimmed_feature_vectors.append(trimmed_dictionary)

    return trimmed_feature_vectors


def write_output_to_file(file_to_create, data_inner, dictionary_inner):

    with open(file_to_create, 'w+') as filehandle:

        for j in range(0, len(data_inner)):

            filehandle.write(data_inner[j].split('\t')[0])

            dict_to_write = dictionary_inner[j]

            for key, value in dict_to_write.items():

                filehandle.write('\t'+ str(key)+':'+str(value))

            if j!=len(data_inner)-1:

                filehandle.write('\n')


if __name__ == "__main__":


    file_train_input = sys.argv[1]

    file_validation_input = sys.argv[2]

    file_test_input = sys.argv[3]

    dictionary_file = sys.argv[4]

    file_train_output = sys.argv[5]

    file_validation_output = sys.argv[6]

    file_test_output = sys.argv[7]

    feature_flag = sys.argv[8]

    input_files = [file_train_input, file_validation_input, file_test_input ]

    output_files = [file_train_output, file_validation_output, file_test_output]

    dictionary = read_dictionary(dictionary_file)

    for i in range(0,3):

        data = read_file(input_files[i])

        if(int(feature_flag) == 1):

            feature_vectors = feature_vector_method_1(data)
            write_output_to_file(output_files[i], data, feature_vectors)

        else:

            feature_vectors_trimmed = feature_vector_method_2(data, dictionary)
            write_output_to_file(output_files[i], data, feature_vectors_trimmed)










