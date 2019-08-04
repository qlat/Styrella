import collections
import math
import os

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
from networkx.readwrite import json_graph

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xlsxwriter

from base_analyzer import *
import measures

class DeltaAnalyser(BaseAnalyzer):

    # Most frequent words in the whole corpus
    corpus_mfw = {}

    # Relative frequencies of words for each text
    texts_rel_freqs = {}

    corpus_features = {}

    mean_values = {}
    z_scores = {}

    strong_edge_weights = [15.0, 5.0, 1.0]
    weak_edge_weights = [0.1, 0.1, 0.1]

    mfw_types = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]

    mfw = 700

    culling_types = [0, 0.3, 0.5, 0.7, 0.9, 1]

    cull = 0

    workbook = None

    measure = ('cosine', measures.delta_measures['cosine'])
    ap_matrix = None
    selected_index = None


    def get_progress_max(self):

        progress_max = 0

        if self.mode == 'auto':
            # auto_select_paramters()
            # get_text_rel_freqs()
            progress_max += len(self.corpus_reader.corpus) * 3
            #get_mfw_in_corpus()
            progress_max += len(self.corpus_reader.corpus) * 3

            progress_max += len(measures.delta_measures) * len(self.mfw_types) * len(self.culling_types)

        # get_mfw_in_corpus()
        progress_max += len(self.corpus_reader.corpus)
        # get_text_rel_freqs()
        progress_max += len(self.corpus_reader.corpus)
        # make_graph()
        progress_max += len(self.corpus_reader.corpus) * 2 # create nodes, ranking
        progress_max += len(self.mfw_types) * len(self.culling_types)
        print('Progress max: '+str(progress_max))
        return progress_max


    def set_delta_measure(self, measure):
        self.measure = (measure, measures.delta_measures[measure])


    def get_mfw_in_corpus(self, corpus):

        """
        Returns a dictionary of the most frequent words
            in the given corpus.
        :param corpus:
        :return:
        """

        print('Getting most frequent words in the whole corpus...')

        # Merge all texts into one big list
        united_corpus = []
        for text in corpus:
            united_corpus.extend(corpus[text])

            self.progress_indicator.inc()

        # Get a ranked list of the absolute frequencies
        sorted_by_freq = [x for x, y in collections.Counter(united_corpus).most_common()]

        return sorted_by_freq


    def get_texts_rel_freqs(self, corpus):
        """ Calculates the relative word frequencies for each text in the given corpus.

            Parameters
            ----------
            corpus : dictionary
                The corpus containing the texts.

        """

        print('Getting the relative word frequencies for each text...')

        result = {}

        for text in corpus:

            len_text = len(corpus[text])
            result[text] = {}

            for x, y in collections.Counter(corpus[text]).most_common(1000):
                result[text][x] = y / len_text

            self.progress_indicator.inc()

        return result


    def reduce_texts_to_mfw(self, texts, mfw):

        result = {}

        for text in texts:

            # Get mfw most frequent words
            reduced_list = list(texts[text].items())[0:mfw]
            reduced_dict = dict([(x, y) for x, y in reduced_list])

            result[text] = reduced_dict

        return result



    def build_word_list(self, texts):

        result = {}

        for text in texts:

            #print('Length of words in >'+text+'< : '+str(len(texts[text])))
            result.update(texts[text])

        # Get mfw words from global corpus
        sorted_list = [(x, y) for x, y in collections.Counter(result).most_common()]

        return [x for x,y in sorted_list]


    def cull_words(self, texts, corpus_word_list, cull_at):

        """
        Creates a subset of the words in 'corpus_word_list' that occur at leat
        in 'cull_at' percent of all 'texts'.

        :param texts:
        :param corpus_word_list:
        :param cull_at:
        :return: List of words from 'corpus_word_list' that occur at least in 'cull_at'
                 percent of all texts.
        """

        len_corpus = len(texts)

        words_to_cull = []
        n_culled = 0

        result = []

        for word in corpus_word_list:

            # Get the number of texts that have the word 'word' in them
            n = 0
            for text in texts:
                if word in texts[text]:
                    n += 1

            # Omit this word if it is not frequent enough
            if (n / len_corpus) < cull_at:
                words_to_cull.extend([word])
                n_culled += 1
            else:
                result.extend([word])

        return result



    def cull_texts(self, texts, corpus_word_list, cull_at):

        len_corpus = len(texts)

        words_to_cull = []
        n_culled = 0

        new_corpus = []

        for word in corpus_word_list:

            new_corpus.extend([word])

            n = 0
            for text in texts:
                if word in texts[text]:
                    n += 1

            if (n / len_corpus) < cull_at:

                words_to_cull.extend([word])
                n_culled += 1


        new_corpus = [w for w in corpus_word_list if not w in words_to_cull]

        for word in words_to_cull:
            for text in texts:

                if word in texts[text]:
                    del texts[text][word]

        return new_corpus


    def get_z_scores(self, texts, corpus_words, mfw, cull_at):

        """
        Calculates the z-scores for the given texts.

        Assumes there are at least mfw common words in all texts.

        :param texts:
        :param mfw:
        :param cull_at:
        :return: Pandas dataframe (rows: texts, columns: words) with z-scores.
        """

        culled_corpus_word_list = self.cull_words(texts, corpus_words, cull_at)

        if mfw > len(culled_corpus_word_list):
            true_mfw = len(culled_corpus_word_list)
        else:
            true_mfw = mfw

        final_word_list = culled_corpus_word_list[:true_mfw]

        # Get the relative frequencies for each text
        matrix = np.zeros((len(texts), true_mfw))

        row = 0
        col = 0
        for text in texts:

            col = 0

            features = np.zeros(true_mfw)

            for word in final_word_list:
                if word in texts[text]:
                    features[col] = texts[text][word]
                else:
                    features[col] = 0

                col += 1

            matrix[row] = features
            row += 1

        scaler = StandardScaler()
        trans = scaler.fit_transform(matrix)

        index = [text for text in texts]
        columns = final_word_list

        return pd.DataFrame(trans, index=index, columns=columns), true_mfw


    def get_ranking(self, text, texts, z_scores, measure):

        result = {}

        for compare_text in texts:
            if not text == compare_text:

                #result[compare_text] = measure(texts[text], texts[compare_text], z_scores.loc[text].values, z_scores.loc[compare_text].values)
                result[compare_text] = measure(z_scores.loc[text].values, z_scores.loc[compare_text].values)

        result = sorted(result.items(), key=lambda x: x[1])
        return result



    def make_graph(self):

        G = nx.DiGraph()

        measure = measures.delta_measures[self.measure[0]]

        z_scores, true_mfw = self.get_z_scores(self.texts_rel_freqs, self.corpus_mfw, self.mfw, self.cull)
        self.z_scores = z_scores

        print(self.z_scores)


        overall = len(self.texts_rel_freqs) + len(self.mfw_types) * len(self.culling_types) * len(self.texts_rel_freqs)
        n_passes = 0

        # Add nodes for all texts forefront
        for text in self.texts_rel_freqs:
            G.add_node(text)
            self.progress_indicator.inc()

        # Choosen settings => strong weights
        for text in self.texts_rel_freqs:

            ranking = self.get_ranking(text, self.texts_rel_freqs, z_scores, measure)

            self.add_edges(G, text, ranking, self.strong_edge_weights)

            print('Creating graph... [{:.2%}]\r'.format(n_passes / overall), sep=' ', end='', flush=True)
            n_passes += 1

            self.progress_indicator.inc()



        # Cycle through settings => weak edge weights
        for mfw in self.mfw_types:

            # Check whether we so many words at all
            if mfw > len(self.corpus_mfw):
                break

            for cull_at in self.culling_types:

                # Don't do main settings again
                if not (mfw == self.mfw and cull_at == self.cull):

                    #z_scores = self.get_z_scores(self.mf_words, mfw, cull_at)
                    z_scores, true_mfw = self.get_z_scores(self.texts_rel_freqs, self.corpus_mfw, mfw, cull_at)

                    for text in self.texts_rel_freqs:

                        ranking = self.get_ranking(text, self.texts_rel_freqs, z_scores, measure)

                        self.add_edges(G, text, ranking, self.weak_edge_weights)

                        print('Creating graph... [{:.2%}]\r'.format(n_passes / overall), sep=' ', end='', flush=True)
                        n_passes += 1

                self.progress_indicator.inc()

        print()

        # Make a copy of edges to prevent errors during iteration
        edges = [e for e in G.edges(data=True)]

        # Remove edges with low connections strength
        for edge in edges:

            if edge[2]['weight'] <= 0.5:

                # Remove edge from graph
                G.remove_edge(edge[0], edge[1])

                # Remove outgoing connection
                if 'Out' in G.nodes[edge[0]]:
                    G.nodes[edge[0]]['Out'] -= 1

                # Remove incoming connection
                if 'In' in G.nodes[edge[0]]:
                    G.nodes[edge[0]]['In'] -= 1

                print('Remove edge from >' + edge[0] + '< to >' + edge[1] + '<')

        return G


    def auto_select_parameters(self):

        print('Auto-selecting parameters...')

        # Temporary corpus for testing
        self.progress_indicator.set_label('[b]Burrow\'s Delta:[/b]\nPreparing test corpus...')
        test_word_freqs = self.get_texts_rel_freqs(self.corpus_reader.test_corpus)
        test_mfw_corpus = self.get_mfw_in_corpus(self.corpus_reader.test_corpus)

        n_measures = len(measures.delta_measures)
        n_mfw = len(self.mfw_types)
        n_culling_types = len(self.culling_types)

        matrix = np.zeros(shape=(n_measures, n_mfw, n_culling_types))
        measures = list(measures.delta_measures.keys())
        print(measures)

        self.progress_indicator.set_label('[b]Burrow\'s Delta:[/b]\nCalculating AP values...')
        n_passes = 0
        for measure in measures.delta_measures:

            i = measures.index(measure)

            for mfw_type in self.mfw_types:

                j = self.mfw_types.index(mfw_type)

                for culling_type in self.culling_types:

                    k = self.culling_types.index(culling_type)

                    links = {}
                    l = 0

                    z_scores, true_mfw = self.get_z_scores(test_word_freqs, test_mfw_corpus, mfw_type, culling_type)

                    not_enough_words = False
                    if true_mfw >= mfw_type:

                        for text in test_word_freqs:

                            ranking = self.get_ranking(text, test_word_freqs, z_scores, measures.delta_measures[measure])

                            relevant = text[:4] == ranking[0][0][:4]

                            link = (text, ranking[0][0], ranking[0][1], relevant)

                            links[l] = link
                            l += 1
                    else:
                        not_enough_words = True


                    sort = sorted(links.items(), key=lambda x: x[1][2])

                    relevance_array = [x[1][3] for x in sort]

                    if not_enough_words:
                        ap = -1
                    else:
                        ap = self.average_precision(relevance_array)

                    matrix[i, j, k] = ap

                    n_passes += 1
                    percentage = n_passes / (n_measures * n_mfw * n_culling_types)
                    print('Calculating AP values [{:.2%}]\r'.format(percentage), sep=' ', end='', flush=True)

                    self.progress_indicator.inc()



        print('')
        self.ap_matrix = matrix

        # Find index of maximum value from 2D numpy array
        choosen_index = np.unravel_index(matrix.argmax(), matrix.shape)

        print('AP matrix:')
        print(matrix)


        print('=> Choose parameters:')
        print('Delta measure: ' + str(measures[choosen_index[0]]))
        print('MFWs: ' + str(self.mfw_types[choosen_index[1]]))
        print('Culling at: ' + str(self.culling_types[choosen_index[2]]))

        self.selected_index = choosen_index

        self.measure = (measures[choosen_index[0]], measures.delta_measures[measures[choosen_index[0]]])
        self.mfw = self.mfw_types[choosen_index[1]]
        self.cull = self.culling_types[choosen_index[2]]



    def analyse(self):

        self.progress_indicator.new_task(self.progress_max)

        if self.mode == 'auto':

            self.auto_select_parameters()

        print('Analysing corpus using the following paramters:')
        print('Delta measure: ' + self.measure[0])
        print('MFWs: ' + str(self.mfw))
        print('Culling: ' + str(self.cull))

        # Get the most frequent words in the whole corpus
        self.progress_indicator.set_label('[b]Burrow\'s Delta:[/b]\nGetting most frequent words...')
        self.corpus_mfw = self.get_mfw_in_corpus(self.corpus_reader.corpus)

        # Calculate the relative word frequencies for each text
        self.progress_indicator.set_label('[b]Burrow\'s Delta:[/b]\nGetting relative frequencies...')
        self.texts_rel_freqs = self.get_texts_rel_freqs(self.corpus_reader.corpus)

        self.progress_indicator.set_label('[b]Burrow\'s Delta:[/b]\nCreating graph...')
        self.graph = self.make_graph()



    def write_values(self):

        worksheet = self.workbook.add_worksheet('Values')

        n_col = 0
        n_row = 2
        n_word = 1

        for word in self.z_scores.columns:
            # Write number of word
            worksheet.write(n_row, n_col, str(n_word))
            # Write word
            worksheet.write(n_row, n_col+1, word)

            n_word += 1
            n_row += 1

            if n_word > 100:
                break

        n_col += 2
        n_row = 0

        # Write values for each text
        for text in self.z_scores.index:

            # Write text name, span over to cells
            worksheet.merge_range(n_row, n_col, n_row, n_col+1, text)

            n_row += 1

            # Write 'rel. freq.' and 'z-score'
            worksheet.write(n_row, n_col, 'rel. freq.')
            worksheet.write(n_row, n_col+1, 'z-score')

            n_row += 1
            n_word = 1
            #for word in self.z_scores[text]:
            for word in self.z_scores.columns:

                freq = 0

                if word in self.texts_rel_freqs[text]:
                    freq = self.texts_rel_freqs[text][word]

                worksheet.write(n_row, n_col, "{0:.3f}".format(round(freq, 3)))
                worksheet.write(n_row, n_col + 1, "{0:.3f}".format(round(self.z_scores.at[text,word], 3)))

                n_word += 1
                n_row += 1

                if n_word > 100:
                    break


            n_row = 0
            n_col += 2


    def write_pca(self):

        print('Writing PCA...')

        pca = PCA(n_components=2)

        principal_components = pca.fit(self.z_scores.to_numpy())
        trans = principal_components.transform(self.z_scores.to_numpy())

        X = trans[:, 0]
        Y = trans[:, 1]

        sns.set_style('white')

        plt.figure(figsize=(8, 6))

        plot = sns.scatterplot(X, Y, color='#D9D9D9')
        for p in zip(X, Y, [text for text in self.z_scores.index]):
            plot.text(p[0]+0.2, p[1]+0.2, p[2], size='8', horizontalalignment='center', verticalalignment='top')

        plot.text(x=0.5, y=1.1, s='2 component PCA (Delta)', fontsize=14, weight='bold', ha='center', va='bottom', transform=plot.transAxes)
        plot.text(x=0.5, y=1.05, s='Delta measure: '+self.measure[0].capitalize()+', mfw: '+str(self.mfw)+', culling at '+str(self.cull*100)+'%', fontsize=8, alpha=0.75, ha='center', va='bottom', transform=plot.transAxes)

        plot.set_xlabel('Principle component 1', fontsize=10)
        plot.set_ylabel('Principle component 2', fontsize=10)

        fig = plot.get_figure()
        fig.savefig(self.output_dir / 'delta_pca.png')
        fig.savefig(self.output_dir / 'delta_pca.svg')



    def write_ap_values(self):

        print('Writing AP values...')

        worksheet = self.workbook.add_worksheet('AP values')

        bold = self.workbook.add_format({'bold': True, 'bg_color': '#DEDEDE'})
        regular = self.workbook.add_format({'bold': False})

        n_row = 0

        # Write headline
        headline = [''] + ['cull '+str(x*100)+'%' for x in self.culling_types]
        for n_col in range(len(headline)):
            worksheet.write(n_row, n_col, headline[n_col])

        n_row += 1

        for i in range(self.ap_matrix.shape[0]):
            for j in range(self.ap_matrix.shape[1]):

                measure = list(measures.delta_measures.keys())[i]
                mfw_type = self.mfw_types[j]

                row = [measure + ' (top '+str(mfw_type) + ' words)']

                for k in range(self.ap_matrix.shape[2]):

                    # This is the actual ap value
                    row.append("{0:.3f}".format(round(self.ap_matrix[i,j,k], 3)))

                for n_col in range(len(row)):

                    if (i,j,n_col-1) == self.selected_index:
                        worksheet.write(n_row, n_col, row[n_col], bold)
                    else:
                        worksheet.write(n_row, n_col, row[n_col], regular)

                n_row += 1



    def write_results(self, output_dir):

        self.output_dir = output_dir / 'delta'
        os.mkdir(self.output_dir)

        print('Write results to >' + str(self.output_dir) + '<')

        self.progress_indicator.set_label('[b]Burrow\'s Delta:[/b]\nWriting results...')

        # Write Excel file
        self.workbook = xlsxwriter.Workbook(self.output_dir / 'delta_data.xlsx')

        self.write_values()

        if self.ap_matrix is not None:
            self.write_ap_values()

        self.workbook.close()

        # Write graph to GraphML file
        nx.write_graphml(self.graph, self.output_dir / 'delta_graph.graphml')

        # Write graph to GEXF file
        nx.write_gexf(self.graph, self.output_dir / 'delta_graph.gexf')

        # Write graph to node-link JSON
        # See https://networkx.github.io/documentation/stable/reference/readwrite/json_graph.html
        data = json_graph.node_link_data(self.graph)

        with open(self.output_dir / 'delta_graph.json', 'w') as f:
            json.dump(data, f)

        self.write_pca()

        print('Finished writing results.')