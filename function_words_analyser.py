import csv
import datetime
from heapq import nlargest
import math
import os
from pathlib import Path
import time

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
import xlsxwriter

import function_words_features
import measures


class FeatureAnalyser:

    feature_choice = {
        'alius': False,
        'antequam': False,
        'conditional': True,
        'conjunctions': True,
        'cum': False,
        'demonstrative': True,
        'dum': False,
        'gerunds': False,
        'idem': False,
        'indefinite': False,
        'ipse': True,
        'iste': False,
        'personal': True,
        'priusquam': False,
        'quidam': False,
        'quin': False,
        'reflexive': False,
        'relative': True,
        'superlatives': False,
        'ut': True
    }

    most_important_features = None

    output_dir = ''

    # Keys: text file names
    # Values: list of words
    corpus_reader = None

    # Possible values: auto, manual
    mode = 'auto'

    # Feature values from each text. Rows: text names; Columns: features; Entries: relative freqs
    feature_matrix = None

    # Distance matrix with selected features only
    distance_matrix_sel = None

    # Selected distance measure
    distance_measure = ('matusita', measures.distance_measures['matusita'])

    graph = None

    strong_edge_weights = [12.0, 6.0, 3.0, 1.5, 0.75]
    weak_edge_weights = [3.0, 2.0, 1.0, 0.25, 0.1, 0.1]

    workbook = None
    
    # Ranked list of ap values for each distance measure
    ap_list = None

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

        # get_feature_values(): #corpus
        progress_max = len(self.corpus_reader.corpus)

        if self.mode == 'auto':
            # auto_select_features(): very low
            progress_max += 1

            # auto_select_distance_measure(): #measures * #test_corpus
            progress_max += len(self.corpus_reader.corpus) * 3 # get_feature_values on test corpus
            progress_max += len(self.corpus_reader.corpus) * 3 * len(measures.distance_measures)

        # make_distance_matrix(): #texts^2
        progress_max += len(self.corpus_reader.corpus) * len(self.corpus_reader.corpus)

        # make_graph()
        progress_max += 3 * len(self.corpus_reader.corpus)

        return progress_max


    def auto_select_features(self, n=7):

        print('Choosing best features...')

        #self.task_label.text = 'Function words: Selecting features...'
        self.progress_indicator.set_label('[b]Function words:[/b]\nSelecting features...')

        # Feature extraction
        test = SelectKBest(score_func=chi2, k=4)

        X = self.feature_matrix
        # Text file names as class labels
        Y = [t for t in self.corpus_reader.corpus]

        fit = test.fit(X, Y)

        indices = nlargest(7, range(len(fit.scores_)), key=lambda idx: fit.scores_[idx])
        self.most_important_features = [1 if i in indices else 0 for i in range(len(fit.scores_))]

        print('Auto select features:')
        for i in indices:
            f = list(self.feature_choice.keys())[i]
            print('  >' + f + '<')
            self.feature_choice[f] = True

        self.progress_indicator.inc()


    def get_feature_count(self, word_list, feature):

        # Source: https://stackoverflow.com/questions/26663371/python-intersection-of-two-lists-keeping-duplicates
        items = set(feature)
        found = [i for i in word_list if i in items]

        return len(found) / len(word_list)


    def get_cum_clauses_count(self, word_list):

        count = 0
        found_cum = False

        for w in word_list:

            if w.lower() == 'cum':
                found_cum = True
            else:
                if found_cum:

                    if not any((w.endswith(s) for s in function_words_features.cumClauses)):
                        count = count + 1

                    found_cum = False

        return count / len(word_list)


    def get_feature_count_substring(self, word_list, feature):

        # Source:
        # https://stackoverflow.com/questions/45926727/python-list-of-substrings-in-list-of-strings

        found = sum(any(m in L for m in feature) for L in word_list)

        return found / len(word_list)



    def get_feature_values(self, corpus, choice):

        # Number of texts
        n_rows = len(corpus)
        n_cols = sum(f == True for f in choice)

        feature_matrix = np.ones(shape=(n_rows, n_cols))

        len_corpus = len(corpus)

        i = 0
        for text in corpus:
            # Index into choice array, usually not the same as k!
            j = 0
            k = 0

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.alius)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.anteq)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.conditionalClauses)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.conjunctions)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_cum_clauses_count(corpus[text])
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.demonstrativePronoun)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.dum)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count_substring(corpus[text], function_words_features.gerund)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.idem)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.indef)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.ipse)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.iste)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.personalPronoun)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.priu)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.quidam)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.quin)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.reflexivePronoun)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.relatives)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count_substring(corpus[text], function_words_features.superlatives)
                j += 1
            k += 1

            if choice[k]:
                feature_matrix[i][j] = self.get_feature_count(corpus[text], function_words_features.ut)
            k += 1

            i += 1

            percentage = i / len_corpus
            print('Getting feature values [{:.2%}]\r'.format(percentage), sep=' ', end='', flush=True)

            self.progress_indicator.inc()

        print()
        print('get_feature_values: '+str(i)+' incs.')

        return feature_matrix


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


    def auto_select_distance_measure(self):

        self.progress_indicator.set_label('[b]Function words:[/b]\nSelecting distance measure...')

        # Build test corpus if it does not already exist
        if len(self.corpus_reader.test_corpus) == 0:
            self.corpus_reader.test_corpus = self.build_test_corpus()

        print('Choosing best distance measure...')
        matrix = self.get_feature_values(self.corpus_reader.test_corpus, [self.feature_choice[x] for x in self.feature_choice])

        links = {}
        texts = [t for t in self.corpus_reader.test_corpus]
        ap = {}

        for measure in measures.distance_measures:

            for i in range(matrix.shape[0]):

                self.distance_measure = (measure, measures.distance_measures[measure])

                # Get Ranking of texts for selected distance measure
                ranking = self.get_ranking(i, matrix, self.distance_measure[1])

                # Link is relevant if both chunks come from same text
                relevant = texts[i][:4] == texts[ranking[0][0]][:4]

                links[i] = ranking[0] + (relevant,)

                self.progress_indicator.inc()


            # Sort descending by distance, needed for prec precision measure
            sort = sorted(links.items(), key=lambda x: x[1][1])

            # Get AP for relevance array
            relevance_array = [x[1][2] for x in sort]
            ap[measure] = self.average_precision(relevance_array)

        self.ap_list = sorted(ap.items(), key=lambda x: x[1], reverse=True)

        print('AP values for distance measures:')
        for dm in self.ap_list:
            print(dm)

        # Choose the first distance measure, i.e. distance measure with highest ap value
        print('Choose distance measure >'+self.ap_list[0][0]+'< (AP='+str(self.ap_list[0][1])+').')
        self.distance_measure = (self.ap_list[0][0], measures.distance_measures[self.ap_list[0][0]])


    def make_distance_matrix(self, matrix):

        print('Calculating distance matrix...')
        self.progress_indicator.set_label('[b]Function words:[/b]\nCalculating distances...')

        dim = matrix.shape[0]
        result = np.zeros(shape=(dim, dim))

        row = 0
        column = 0

        for i in range(dim):
            for j in range(dim):

                if not i == j:
                    result[i,j] = self.distance_measure[1](self.feature_matrix[i,:], self.feature_matrix[j,:])
                else:
                    result[i,j] = 0
                column += 1

                self.progress_indicator.inc()

            column = 0
            row +=1

        return result


    def get_ranking(self, n_row, texts, matrix, measure):

        a = {}

        # Loop through rows
        for row in range(matrix.shape[0]):

            if not n_row == row:
                # Get distance from n_row to current row
                a[texts[row]] = measure(matrix[n_row, :], matrix[row, :])

        a_sorted = sorted(a.items(), key=lambda x: x[1])

        return a_sorted



    def add_edges(self, graph, label, ranking, weights):

        # Add edges to first and two runner-ups
        for i in range(len(weights)):
            link_label = ranking[i][0]

            # Create/increase in degree
            if 'In' in graph.nodes[link_label]:
                graph.nodes[link_label]['In'] += 1
            else:
                graph.nodes[link_label]['In'] = 1

            if not graph.has_edge(label, link_label):
                graph.add_edge(label, link_label)
                graph[label][link_label]['weight'] = weights[i]

                # Increase out degree only if this is a new edge
                if 'Out' in graph.nodes[label]:
                    graph.nodes[label]['Out'] += 1
                else:
                    graph.nodes[label]['Out'] = 1
            else:
                graph[label][link_label]['weight'] += weights[i]


    def make_graph(self, feature_matrix, text_labels, selected_features):

        print('Creating graph...')
        self.progress_indicator.set_label('[b]Function words:[/b]\nCreating graph...')

        G = nx.DiGraph()

        # Filter selected features
        selection = np.array(selected_features)
        matrix = feature_matrix[:, selection]

        # Add nodes for all texts forefront
        for i in range(matrix.shape[0]):
            G.add_node(text_labels[i])

            self.progress_indicator.inc()

        # Step 1: Compare selected features at once => strong edge weights
        for i in range(matrix.shape[0]):

            ranking = self.get_ranking(i, text_labels, matrix, self.distance_measure[1])

            self.add_edges(G, text_labels[i], ranking, self.strong_edge_weights)
            self.progress_indicator.inc()


        # Step 2: Compare all features at once => weak edge weights
        for i in range(feature_matrix.shape[0]):

            ranking = self.get_ranking(i, text_labels, self.feature_matrix, self.distance_measure[1])

            self.add_edges(G, text_labels[i], ranking, self.weak_edge_weights)
            self.progress_indicator.inc()


        return G


    def analyse(self):

        # Get values for all features
        self.progress_indicator.new_task(self.progress_max)
        self.progress_indicator.set_label('[b]Function words:[/b]\nGetting values...')

        self.feature_matrix = self.get_feature_values(self.corpus_reader.corpus, [1 for x in range(len(self.feature_choice))])


        if self.mode == 'auto':

            # Unchoose all features
            for key in self.feature_choice:
                self.feature_choice[key] = False

            # Auto-select features
            self.auto_select_features()

            # Auto-select distance measure
            self.auto_select_distance_measure()

        if self.mode == 'manual':
            pass

        print('Analysing corpus using the following paramters:')
        print('Use features:')
        for f in self.feature_choice:
            if self.feature_choice[f]:
                print('>'+f+'<')
        print('Distance measure: '+self.distance_measure[0])

        # Make distance matrix for selected features only
        selection = [self.feature_choice[f] for f in self.feature_choice]
        matrix = self.feature_matrix[:, selection]

        self.distance_matrix_sel = self.make_distance_matrix(matrix)

        # Make graph
        self.G = self.make_graph(self.feature_matrix, [t for t in self.corpus_reader.corpus], [self.feature_choice[f] for f in self.feature_choice])


    def write_features(self):

        print('Writing feature values...')

        worksheet = self.workbook.add_worksheet('Feature values')

        n_row = 0
        n_features = self.feature_matrix.shape[1]

        # Define format for selected features
        bold = self.workbook.add_format({'bold': True})
        regular = self.workbook.add_format({'bold': False})
        formats = [regular for i in range(n_features+1)]

        selection = [self.feature_choice[f] for f in self.feature_choice]
        formats[0] = regular

        # Apply format for selected features
        for i in range(n_features):
            if selection[i]:
                formats[i+1] = bold

        # Write headline
        headline = [''] + [x for x in self.feature_choice]

        for n_col in range(n_features + 1):
            worksheet.write(n_row, n_col, headline[n_col], formats[n_col])

        n_row += 1

        # Write values
        texts = [t for t in self.corpus_reader.corpus]

        for i in range(self.feature_matrix.shape[0]):
            row = [texts[i]] + list(self.feature_matrix[i,:])

            # Write first item
            worksheet.write(n_row, 0, row[0], formats[0])

            for n_col in range(1, n_features + 1):
                worksheet.write(n_row, n_col, "{0:.3f}".format(round(row[n_col], 3)), formats[n_col])
            n_row += 1

        worksheet.write(n_row+1, 0, 'Selected features in bold.')


    def write_distances(self):

        print('Writing distance values...')

        worksheet = self.workbook.add_worksheet('Distances')

        n_row = 0
        n_distances = self.distance_matrix_sel.shape[0]

        # Write headline
        texts = [t for t in self.corpus_reader.corpus]
        headline = [''] + texts
        for n_col in range(n_distances + 1):
            worksheet.write(n_row, n_col, headline[n_col])

        n_row += 1

        # Write values
        for i in range(n_distances):
            row = [texts[i]] + list(self.distance_matrix_sel[i,:])

            # Write first item (text only => do not format)
            worksheet.write(n_row, 0, row[0])

            for n_col in range(1, n_distances + 1):
                worksheet.write(n_row, n_col, "{0:.3f}".format(round(row[n_col], 3)))
            n_row += 1

        worksheet.write(n_row + 1, 0, 'Distance measure: '+self.distance_measure[0])


    def write_ap_values(self):

        worksheet = self.workbook.add_worksheet('AP values')

        bold = self.workbook.add_format({'bold': True, 'bg_color': '#DEDEDE'})
        regular = self.workbook.add_format({'bold': False})

        n_row = 0

        worksheet.write(n_row, 0, 'Distance measure')
        worksheet.write(n_row, 1, 'AP')
        n_row += 1

        temp_dict = dict(self.ap_list)

        for ap in measures.distance_measures:

            format = regular
            if ap == self.distance_measure[0]:
                format = bold

            worksheet.write(n_row, 0, ap, regular)
            worksheet.write(n_row, 1, "{0:.3f}".format(round(temp_dict[ap], 3)), format)

            n_row += 1


    def write_pca(self):

        print('Writing PCA...')

        # Do z-normalization
        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(self.feature_matrix)

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

        plot.text(x=0.5, y=1.1, s='2 component PCA (function words)', fontsize=14, weight='bold', ha='center', va='bottom',
                  transform=plot.transAxes)
        plot.text(x=0.5, y=1.05,
                  s='Distance measure: ' + self.distance_measure[0].capitalize(),
                  fontsize=8, alpha=0.75, ha='center', va='bottom',
                  transform=plot.transAxes)

        plot.set_xlabel('Principle component 1', fontsize=10)
        plot.set_ylabel('Principle component 2', fontsize=10)

        plt.tight_layout()

        fig = plot.get_figure()
        fig.savefig(self.output_dir / 'function_words_pca.png')
        fig.savefig(self.output_dir / 'function_words_pca.svg')

        
    def write_results(self, output_dir):

        self.output_dir = output_dir / 'function_words'
        os.mkdir(self.output_dir)

        print('Write results to >'+str(self.output_dir)+'<')

        self.write_pca()

        # Write Excel file
        self.workbook = xlsxwriter.Workbook(self.output_dir / 'function_words_data.xlsx')

        # Write feature values to Excel file
        self.write_features()

        # Write distances to Excel file
        self.write_distances()

        # Write test values to Excel file
        if self.ap_list is not None:
            self.write_ap_values()

        self.workbook.close()

        # Write graph to GraphML file
        nx.write_graphml(self.G, self.output_dir / 'function_words_graph.graphml')

        # Write graph to GEXF file
        nx.write_gexf(self.G, self.output_dir / 'function_words_graph.gexf')

        # Write graph to node-link JSON
        # See https://networkx.github.io/documentation/stable/reference/readwrite/json_graph.html
        data = json_graph.node_link_data(self.G)

        with open(self.output_dir / 'function_words_graph.json', 'w') as f:
            json.dump(data, f)

        print('Finished writing results.')
