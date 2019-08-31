import chardet
import os
from pathlib import Path
from re import findall

class CorpusReader:

    # Keys: text file names
    # Values: list of words
    corpus = {}

    # Texts divided into four chunks
    test_corpus = {}

    # Number of chunks to split each text into for testing purposes
    n_test_chunks = 3

    def get_encoding(self, filename):

        f = open(filename, 'rb')
        guess = chardet.detect(f.read())

        return guess['encoding']

    def read_file(self, name):

        result = []

        encoding = self.get_encoding(name)
        print('Open >'+name.name+'< with encoding >'+encoding+'<')

        f = open(name, 'r', encoding=encoding)
        for l in f:

            # Convert to lower-case
            line = l.lower()

            # Replace some special symbols
            line = line.replace('á', 'a').replace('é', 'e').replace('ë', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
            line = line.replace("'", '')

            # Replace j by i, v by u
            line = line.replace('j', 'i')
            line = line.replace('v', 'u')

            # Return only words. Source:
            # https://stackoverflow.com/questions/1059559/split-strings-with-multiple-delimiters
            result.extend(findall(r"[\w']+", line))

        return result


    def __init__(self, directory):

        print('Reading corpus...')

        filenames = [f for f in os.listdir(directory)]
        filenames = sorted(filenames, key=lambda s: s.casefold())

        for filename in filenames:
            path_and_name = directory / filename

            short_filename = Path(path_and_name).stem

            self.corpus[short_filename] = self.read_file(path_and_name)
            print('  ' + short_filename + ' (' + str(len(self.corpus[short_filename])) + ' words)')

        self.test_corpus = self.build_test_corpus()


    def split(self, a, n):

        # Source: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
        # (modified)
        k, m = divmod(len(a), n)
        return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


    def build_test_corpus(self):

        result = {}

        n_text = 1
        n_chunks = len(self.corpus) * self.n_test_chunks
        for t in self.corpus:

            prefix = "{:04d}".format(n_text)

            n_chunk = 1
            # Split text into non-overlapping chunks
            for chunk in self.split(self.corpus[t], self.n_test_chunks):

                if len(chunk) > 100:
                    chunk_name = prefix + '_' + str(n_chunk) + '_' + t
                    result[chunk_name] = chunk

                    percentage = n_chunk / n_chunks
                    print('Building test corpus [{:.2%}]\r'.format(percentage), sep=' ', end='', flush=True)

                    n_chunk += 1

            n_text += 1

        print('Building test corpus [{:.2%}]\r'.format(1), sep=' ', end='', flush=True)
        print()

        return result