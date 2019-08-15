import collections
import datetime
import json
import os

import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import seaborn as sns
import xlsxwriter

import measures

class NGramAnalyser:

    # Keys: text file names
    # Values: list of words
    corpus_reader = None

    output_dir = ''

    # Possible values: auto, manual
    mode = 'auto'

    studied_text = None

    # The 'n' in 'n-gram'
    n = 4

    # Keys: 2...10
    n_grams = {}

    # N-gram parameters
    mf_ngrams = 200
    ngram_types = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    mf_ngram_types = [50, 100, 200, 300, 400, 500, 1000]

    ap_matrix = None
    choosen_index = None


    # Selected distance measure, tupel: (name, function)
    distance_measure = ('tanimoto', measures.distance_measures['tanimoto'])

    distance_matrix = None

    strong_edge_weights = [15.0, 5.0, 1.0]
    weak_edge_weights = [0.1, 0.1, 0.1]

    graph = None

    # Progress indicator
    progress_indicator = None

    progress_max = 0


    def __init__(self, corpus_reader, output_dir, mode):
        self.corpus_reader = corpus_reader
        self.output_dir = output_dir

        self.mode = mode

        self.progress_max = self.get_progress_max()


    def set_distance_measure(self, measure):
        self.distance_measure = (measure, measures.distance_measures[measure])


    def get_progress_max(self):

        progress_max = 0

        if self.mode == 'auto':
            # auto_select_parameters

            # Create test n-grams
            progress_max += len(self.ngram_types) * len(self.corpus_reader.corpus) * 3

            # Calculating AP value
            progress_max += len(measures.distance_measures) * len(self.ngram_types) * len(self.mf_ngram_types) * len(self.corpus_reader.corpus) * 3

        # get_ngrams
        progress_max += len(self.ngram_types) * len(self.corpus_reader.corpus)

        # make_distance_matrix
        progress_max += len(self.corpus_reader.corpus) * (len(self.corpus_reader.corpus) - 1)

        # make_graph
        progress_max += len(self.corpus_reader.corpus)
        progress_max += len(self.ngram_types) * len(self.corpus_reader.corpus) * len(self.mf_ngram_types)

        return progress_max


    def get_ngrams(self, words, n):

         # Join words to string, separeted by whitespaces
        s = " ".join(words)

        # Create n-grams from this string
        ngram_list = [s[i:i + n] for i in range(len(s) - n + 1)]

        # Sort descending by frequency, first 1000 entries (max)
        sorted = collections.Counter(ngram_list).most_common(1000)
        # print(sorted)

        # Save relative frequency, convert to dictionary for easy access
        sorted = dict([(x, y / len(words)) for x, y in sorted])

        return sorted


    def get_ranking(self, text, texts, mf_ngrams, measure):

        a = {}

        for compare_text in texts:
            if not text == texts[compare_text]:

                a[compare_text] = self.get_ngram_distance(text, texts[compare_text], mf_ngrams, measure)

        a_sorted = sorted(a.items(), key=lambda x: x[1])

        return a_sorted


    def precision(self, k, array):

        s = 0
        for i in range(k):
            if array[i]:
                s += 1

        return s / k


    def average_precision(self, array):

        ap = 0
        for i in range(len(array)):
            ap += self.precision(i+1, array) * array[i]

        return ap / len(array)


    def auto_select_parameters(self):

        print('Choosing parameters...')

        test_ngrams = {}

        n_calculations = len(self.ngram_types) * len (self.corpus_reader.test_corpus)
        n_calc = 0

        self.progress_indicator.set_label('[b]N-Grams:[/b]\nPreparing test corpus...')

        # Create 2-grams to 10-grams from test corpus
        for n in self.ngram_types:

            test_ngrams[n] = {}

            for text in self.corpus_reader.test_corpus:

                test_ngrams[n][text] = {}

                test_ngrams[n][text] = self.get_ngrams(self.corpus_reader.test_corpus[text], n)

                n_calc += 1

                percentage = n_calc / n_calculations

                print('Creating ' + str(n)+ '-grams [{:.2%}]\r'.format(percentage), sep=' ', end='', flush=True)
                self.progress_indicator.inc()
        print()

        # Create 3D adjacency matrix (measure x n x mf n-grams)
        n_measures = len(measures.distance_measures)
        n_ngram_passes = len(self.ngram_types)
        n_mfngram_passes = len(self.mf_ngram_types)

        matrix = np.zeros(shape=(n_measures, n_ngram_passes, n_mfngram_passes))
        measures_list = list(measures.distance_measures.keys())

        self.progress_indicator.set_label('[b]N-Grams:[/b]\nCalculating AP values...')
        n_passes = 0
        for measure in measures.distance_measures:

            i = measures_list.index(measure)

            for ngram_type in self.ngram_types:

                j = self.ngram_types.index(ngram_type)

                for mf_ngram_type in self.mf_ngram_types:

                    k = self.mf_ngram_types.index(mf_ngram_type)

                    links = {}
                    l = 0

                    for text in test_ngrams[ngram_type]:

                        ranking = self.get_ranking(test_ngrams[ngram_type][text], test_ngrams[ngram_type], mf_ngram_type, measures.distance_measures[measure])

                        # Link is relevant if both chunks come from same text
                        relevant = text[:4] == ranking[0][0][:4]

                        # Make 4-tuple (source, target, distance, relevant)
                        link = (text, ranking[0][0], ranking[0][1], relevant)
                        links[l] = link

                        l += 1
                        self.progress_indicator.inc()

                    sort = sorted(links.items(), key=lambda x: x[1][2])

                    # Get AP for relevance array
                    relevance_array = [x[1][3] for x in sort]

                    ap = self.average_precision(relevance_array)
                    matrix[i,j,k] = ap

                    n_passes += 1
                    percentage = n_passes / (n_measures*n_ngram_passes*n_mfngram_passes)
                    print('Calculating AP values [{:.2%}]\r'.format(percentage), sep=' ', end='', flush=True)

        print('AP matrix:')
        print(matrix)

        self.ap_matrix = matrix

        # Find index of maximum value from 2D numpy array

        self.choosen_index = np.unravel_index(matrix.argmax(), matrix.shape)
        print('Largest index = '+str(self.choosen_index))
        print('=> Using parameters:')
        print('Distance measure: '+str(measures_list[self.choosen_index[0]]))
        print('N-Gram type: '+str(self.ngram_types[self.choosen_index[1]]))
        print('Most frequent n-grams: '+str(self.mf_ngram_types[self.choosen_index[2]]))

        self.distance_measure = (measures_list[self.choosen_index[0]], measures.distance_measures[measures_list[self.choosen_index[0]]])
        self.n = self.ngram_types[self.choosen_index[1]]
        self.mf_ngrams = self.mf_ngram_types[self.choosen_index[2]]


    def get_ngram_distance(self, a_ngrams, b_ngrams, mf_ngrams, measure):

        list_a = []
        list_b = []

        i = 0

        # Consider n-grams that are in a and in b
        for ngram in a_ngrams:
            if ngram in b_ngrams:

                list_a.extend([a_ngrams[ngram]])
                list_b.extend([b_ngrams[ngram]])

                i += 1
                if i == mf_ngrams:
                    break


        # Account for the rare case where there are not enough ngrams
        if i < mf_ngrams:
            d = 1000000
        else:
            d = measure(np.array(list_a), np.array(list_b))

        return d


    def add_edges(self, graph, label, ranking, weights):

        # Add edges to first and two runner-ups
        for i in range(len(weights)):
            #link_label = labels[ranking[i][0]]
            link_label = ranking[i][0]

            # Create/increase in degree
            if 'In' in graph.nodes[link_label]:
                graph.nodes[link_label]['In'] += 1
            else:
                graph.nodes[link_label]['In'] = 1

            if not graph.has_edge(label, link_label):
                graph.add_edge(label, link_label, weight=weights[i])

                # Increase out degree only if this is a new edge
                if 'Out' in graph.nodes[label]:
                    graph.nodes[label]['Out'] += 1
                else:
                    graph.nodes[label]['Out'] = 1
            else:
                graph[label][link_label]['weight'] += weights[i]


    def make_graph(self):

        G = nx.DiGraph()

        # Filter selected features
        self.progress_indicator.set_label('[b]N-Grams:[/b]\nCreating graph...')

        row_labels = [t for t in self.n_grams[self.n]]

        overall = len(self.n_grams[self.n]) + len(self.ngram_types) * len(self.n_grams[self.n]) * len(self.mf_ngram_types)
        n_passes = 0

        # Add nodes for all texts forefront
        for text in self.n_grams[self.n]:
            G.add_node(text)

        # Step 1: Distances with current settings
        for text in self.n_grams[self.n]:

            ranking = self.get_ranking(self.n_grams[self.n][text], self.n_grams[self.n], self.mf_ngrams, self.distance_measure[1])

            #print(ranking)
            G.add_node(text)
            self.add_edges(G, text, ranking, self.strong_edge_weights)

            n_passes += 1
            print('Creating graph... [{:.2%}]\r'.format(n_passes / overall), sep=' ', end='', flush=True)
            self.progress_indicator.inc()


        # Step 2: Vary n and mf_ngrams
        n_edges = 0
        for n in self.ngram_types:

            for text in self.n_grams[n]:

                    for mfg in self.mf_ngram_types:

                        if not (n == self.n and mfg == self.mf_ngrams):
                            ranking = self.get_ranking(self.n_grams[n][text], self.n_grams[n], mfg, self.distance_measure[1])
                            self.add_edges(G, text, ranking, self.weak_edge_weights)
                            n_edges += 1

                        n_passes += 1
                        print('Creating graph... [{:.2%}]\r'.format(n_passes / overall), sep=' ', end='', flush=True)
                        self.progress_indicator.inc()

        print()

        print('Created '+str(n_edges)+' edges.')

        # Make a copy of edges to prevent errors during iteration
        edges = [e for e in G.edges(data=True)]

        # Remove edges with low connections strength
        for edge in edges:

            if edge[2]['weight'] <= 1.0:

                # Remove edge from graph
                G.remove_edge(edge[0], edge[1])

                # Remove outgoing connection
                if 'Out' in G.nodes[edge[0]]:
                    G.nodes[edge[0]]['Out'] -= 1

                # Remove incoming connection
                if 'In' in G.nodes[edge[0]]:
                    G.nodes[edge[0]]['In'] -= 1

                print('Remove edge from >'+edge[0]+'< to >'+edge[1]+'<')

        return G


    def make_distance_matrix(self):

        print('Calculate distance matrix (distance measure='+self.distance_measure[0]+', '+str(self.n)+'-grams, top '+str(self.mf_ngrams)+')')
        self.progress_indicator.set_label('[b]N-Grams:[/b]\nCalculate distances...')

        dim = len(self.n_grams[self.n])
        matrix = np.zeros(shape=(dim, dim))

        i = 0
        j = 0
        for text in self.n_grams[self.n]:
            for compare_text in self.n_grams[self.n]:

                if not i == j:
                    matrix[i,j] = self.get_ngram_distance(self.n_grams[self.n][text], self.n_grams[self.n][compare_text], self.mf_ngrams, self.distance_measure[1])

                j += 1
                self.progress_indicator.inc()

            j = 0
            i += 1


        print('Distance matrix:')
        print(matrix)
        return matrix



    def analyse(self):

        self.progress_indicator.new_task(self.progress_max)


        if self.mode == 'auto':

            self.progress_indicator.set_label('[b]N-Grams:[/b]\nSelecting parameters...')

            self.auto_select_parameters()


        print('Analysing corpus using the following paramters:')
        print('Distance measure: ' + self.distance_measure[0])
        print('N-Gram type: ' + str(self.n))
        print('Most frequent n-grams: ' + str(self.mf_ngrams))

        n_calculations = len(self.ngram_types) * len(self.corpus_reader.corpus)
        n_calc = 1

        # Get n-grams
        self.progress_indicator.set_label('[b]N-Grams:[/b]\nCreating n-grams...')
        for n in self.ngram_types:

            self.n_grams[n] = {}

            for text in self.corpus_reader.corpus:

                percentage = n_calc / n_calculations
                print('Creating ' + str(n)+ '-grams [{:.2%}]\r'.format(percentage), sep=' ', end='', flush=True)
                n_calc += 1

                self.n_grams[n][text] = self.get_ngrams(self.corpus_reader.corpus[text], n)
                self.progress_indicator.inc()
        print()

        self.distance_matrix = self.make_distance_matrix()

        self.graph = self.make_graph()


    def write_ngrams(self):

        print('Writing n-grams...')

        for n in self.n_grams:

            print('  '+str(n)+'-grams...')

            worksheet = self.workbook.add_worksheet(str(n) + '-grams')

            n_row = 0
            n_col = 0

            for text in self.n_grams[n]:
                worksheet.write(n_row, n_col, text)
                n_row += 1

                for ngram in self.n_grams[n][text]:
                    worksheet.write(n_row, n_col, ngram.replace(' ', '_'))
                    worksheet.write(n_row, n_col + 1, "{0:.3f}".format(round(self.n_grams[n][text][ngram], 3)))

                    n_row += 1

                n_row = 0
                n_col += 2


    def write_distances(self):

        print('Writing distance values.')

        worksheet = self.workbook.add_worksheet('Distances')

        n_row = 0
        n_distances = self.distance_matrix.shape[0]

        # Write headline
        texts = [t for t in self.corpus_reader.corpus]
        headline = [''] + texts
        for n_col in range(n_distances + 1):
            worksheet.write(n_row, n_col, headline[n_col])

        n_row += 1

        # Write values
        for i in range(n_distances):
            row = [texts[i]] + list(self.distance_matrix[i,:])

            # Write first item (no value)
            worksheet.write(n_row, 0, row[0])

            for n_col in range(1, n_distances + 1):
                worksheet.write(n_row, n_col, "{0:.3f}".format(round(row[n_col], 3)))
            n_row += 1

        worksheet.write(n_row + 1, 0, 'Distance measure: '+self.distance_measure[0])


    def write_ap_values(self):

        print('Writing AP values...')

        worksheet = self.workbook.add_worksheet('AP values')

        # Define format for selected features
        bold = self.workbook.add_format({'bold': True, 'bg_color': '#DEDEDE'})
        regular = self.workbook.add_format({'bold': False})


        n_row = 0

        # Write headline
        headline = [''] + [str(x) for x in self.mf_ngram_types]
        for n_col in range(len(headline)):
            worksheet.write(n_row, n_col, headline[n_col], regular)

        n_row += 1

        formats = [regular for i in range(self.ap_matrix.shape[2]+1)]

        for i in range(self.ap_matrix.shape[0]):
            for j in range(self.ap_matrix.shape[1]):

                measure = list(measures.distance_measures.keys())[i]
                ngram_type = self.ngram_types[j]

                row = [str(ngram_type) + ' ' + measure]

                formats = [regular for i in range(self.ap_matrix.shape[2] + 1)]

                for k in range(self.ap_matrix.shape[2]):

                    mf_ngrams = self.mf_ngram_types[k]

                    # This is the actual ap value
                    row.append(self.ap_matrix[i,j,k])

                    if (i,j,k) == self.choosen_index:
                        formats[k+1] = bold

                # Write first col (no float)
                worksheet.write(n_row, 0, row[0], formats[0])

                for n_col in range(1, len(row)):
                    worksheet.write(n_row, n_col, "{0:.3f}".format(round(row[n_col], 3)), formats[n_col])

                n_row += 1


    def write_pca(self):

        print('Writing PCA...')

        # Do z-normalization
        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(self.distance_matrix)

        pca = PCA(n_components=2)

        principal_components = pca.fit(scaled)
        trans = principal_components.transform(scaled)

        X = trans[:, 0]
        Y = trans[:, 1]

        sns.set_style('white')

        plt.figure(figsize=(8, 6))

        plot = sns.scatterplot(X, Y, color='#D9D9D9')

        # Add labels to each point
        for p in zip(X, Y, [text for text in self.corpus_reader.corpus]):
            plot.text(p[0]+0.2, p[1]+0.2, p[2], size='8', horizontalalignment='center', verticalalignment='top')

        plot.text(x=0.5, y=1.1, s='2 component PCA (character n-grams)', fontsize=14, weight='bold', ha='center', va='bottom',
                  transform=plot.transAxes)
        plot.text(x=0.5, y=1.05,
                  s=str(self.mf_ngrams) + ' most frequent ' + str(self.n) + '-grams, distance measure: ' +
                    self.distance_measure[0].capitalize(), fontsize=8, alpha=0.75, ha='center', va='bottom',
                  transform=plot.transAxes)

        plot.set_xlabel('Principle component 1', fontsize=10)
        plot.set_ylabel('Principle component 2', fontsize=10)

        plt.tight_layout()

        fig = plot.get_figure()
        fig.savefig(self.output_dir / 'ngram_pca.png')
        fig.savefig(self.output_dir / 'ngram_pca.svg')



    def write_results(self, output_dir):

        self.output_dir = output_dir / 'n-grams'
        os.mkdir(self.output_dir)

        print('Write results to >' + str(self.output_dir) + '<')
        self.progress_indicator.set_label('[b]N-Grams:[/b]\nWriting results...')


        # Write Excel file
        self.workbook = xlsxwriter.Workbook(self.output_dir / 'ngram_data.xlsx')

        self.write_distances()

        if self.ap_matrix is not None:
            self.write_ap_values()

        self.write_ngrams()

        self.workbook.close()

        # Write graph to GraphML file
        nx.write_graphml(self.graph, self.output_dir / 'ngram_graph.graphml')

        # Write graph to GEXF file
        nx.write_gexf(self.graph, self.output_dir / 'ngram_graph.gexf')

        # Write graph to node-link JSON
        # See https://networkx.github.io/documentation/stable/reference/readwrite/json_graph.html
        data = json_graph.node_link_data(self.graph)

        with open(self.output_dir / 'ngram_graph.json', 'w') as f:
            json.dump(data, f)

        self.write_pca()

        print('Finished writing results.')
