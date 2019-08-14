class BaseAnalyzer:

    corpus_reader = None
    output_dir = ''

    # Possible values: auto, manual
    mode = 'auto'

    # Progress
    progress_indicator = None
    progress_max = 0

    graph = None

    def __init__(self, corpus_reader, output_dir, mode):
        self.corpus_reader = corpus_reader
        self.output_dir = output_dir

        self.mode = mode

        self.progress_max = self.get_progress_max()

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

    def analyse(self):
        raise NotImplementedError("Please Implement this method")

    def make_graph(self):
        raise NotImplementedError("Please Implement this method")

    def auto_select_parameters(self):
        raise NotImplementedError("Please Implement this method")

    def get_progress_max(self):
        raise NotImplementedError("Please Implement this method")

    def write_results(self, output_dir):
        raise NotImplementedError("Please Implement this method")

    def write_ap_values(self):
        raise NotImplementedError("Please Implement this method")

    def write_values(self):
        raise NotImplementedError("Please Implement this method")

    def write_pca(self):
        raise NotImplementedError("Please Implement this method")
