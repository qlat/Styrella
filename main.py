import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

import kivy

import os
from pathlib import Path
import threading

from kivy.app import App
from kivy.clock import *
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

from corpus_reader import *
from delta_analyser import *
from function_words_analyser import *
from ngram_analyser import *


class ProgressIndicator:

    progress_bar = None
    overall_progress_bar = None
    label = None

    @mainthread
    def __init__(self, progress_bar, overall_progress_bar, label):
        self.progress_bar = progress_bar
        self.overall_progress_bar = overall_progress_bar
        self.label = label

    @mainthread
    def new_task(self, max):
        self.progress_bar.value = 0
        self.progress_bar.max = max

    @mainthread
    def set_label(self, label):
        self.label.text = label

    @mainthread
    def inc(self):
        self.progress_bar.value += 1
        self.overall_progress_bar.value += 1

    @mainthread
    def set_overall_progress_max(self, max):
        self.overall_progress_bar.max = max


class StyrellaApp(App):

    ana = None
    nana = None
    dana = None

    auto_mode = True

    do_function_words = True
    do_ngrams = True
    do_delta = True

    corpus_dir = None
    output_dir = None

    corpus_reader = None

    def build(self):

        Window.size = (640, 480)

        self.corpus_dir = Path('corpus_africa').resolve()
        self.root.ids.corpus_dir_input.text = self.corpus_dir.as_posix()

        self.output_dir = Path('output').resolve()
        self.root.ids.output_dir_input.text = self.output_dir.as_posix()

    def do_choose_corpus_dir(self):

        Tk().withdraw()
        dirname = filedialog.askdirectory(initialdir=os.getcwd())
        print('Selected: '+dirname)
        self.root.ids.corpus_dir_input.text = dirname

    def do_choose_output_dir(self):

        Tk().withdraw()
        dirname = filedialog.askdirectory(initialdir=os.getcwd())
        print('Selected: '+dirname)
        self.root.ids.output_dir_input.text = dirname

    def do_next(self):

        current = self.root.ids.screen_manager.current

        if current == 'choose_directories':

            problems = False

            self.corpus_dir = Path(self.root.ids.corpus_dir_input.text).resolve()
            print(self.corpus_dir)

            self.output_dir = Path(self.root.ids.output_dir_input.text).resolve()
            print(self.output_dir)

            Tk().withdraw()

            # Check if directory exists
            if not self.corpus_dir.exists():
                problems = True
                messagebox.showerror("Error", "Corpus directory does not exist!")
            else:
                # If it exists, check if it is readable
                if not os.access(self.corpus_dir, os.R_OK):
                    problems = True
                    messagebox.showerror("Error", "Corpus directory is not readable!")

            # Do the same checks with the output dir
            if not self.output_dir.exists():
                problems = True
                messagebox.showerror("Error", "Output directory does not exist!")
            else:
                # Output dir has to be writeable
                if not os.access(self.output_dir, os.W_OK):
                    problems = True
                    messagebox.showerror("Error", "Output directory is not readable!")

            if not problems:

                self.root.ids.screen_manager.current = 'choose_mode'

                # Disable next button
                self.root.ids.next_button.color = (.9, .9, .9, 1)
                self.root.ids.next_button.diabled = True

                # Enable back button
                self.root.ids.back_button.color = (.23, .23, .23, 1)
                self.root.ids.back_button.diabled = False

        if current == 'choose_features':
            self.root.ids.screen_manager.current = 'choose_distance_measures'

        if current == 'choose_distance_measures':
            self.root.ids.screen_manager.current = 'further_options'

        if current == 'choose_methods':

            if self.root.ids.checkbox_function_words.active:
                self.do_function_words = True
            else:
                self.do_function_words = False

            if self.root.ids.checkbox_ngrams.active:
                self.do_ngrams = True
            else:
                self.do_ngrams = False

            if self.root.ids.checkbox_delta.active:
                self.do_delta = True
            else:
                self.do_delta = False

            print('Do function words: '+str(self.do_function_words))
            print('Do ngrams: ' + str(self.do_ngrams))
            print('Do delta: ' + str(self.do_delta))

            if self.do_function_words:
                self.root.ids.screen_manager.current = 'function_words_choose_features'
            else:
                if self.do_ngrams:
                    self.root.ids.screen_manager.current = 'ngrams_choose_n'
                else:
                    if self.do_delta:
                        self.root.ids.screen_manager.current = 'delta_choose_most_frequent_words'

        # Function words ------------------------------------------------------
        if current == 'function_words_choose_features':
            self.root.ids.screen_manager.current = 'function_words_choose_distance_measure'

        if current == 'function_words_choose_distance_measure':

            # No more options for function words. Check other methods.
            if self.do_ngrams:
                self.root.ids.screen_manager.current = 'ngrams_choose_n'
            else:
                if self.do_delta:
                    self.root.ids.screen_manager.current = 'delta_choose_most_frequent_words'
                else:
                    self.start_manual()


        # Character N-Grams ---------------------------------------------------
        if current == 'ngrams_choose_n':
            self.root.ids.screen_manager.current = 'ngrams_choose_most_common_ngrams'

        if current == 'ngrams_choose_most_common_ngrams':
            self.root.ids.screen_manager.current = 'ngrams_choose_distance_measure'

        if current == 'ngrams_choose_distance_measure':

            # No more options for n-grams. Check other methods.
            if self.do_delta:
                self.root.ids.screen_manager.current = 'delta_choose_most_frequent_words'
            else:
                self.start_manual()


        # Burrows' Delta ------------------------------------------------------
        if current == 'delta_choose_most_frequent_words':
            self.root.ids.screen_manager.current = 'delta_choose_culling_factor'


        if current == 'delta_choose_culling_factor':
            self.root.ids.screen_manager.current = 'delta_choose_delta_measure'


        if current == 'delta_choose_delta_measure':

            # Finished => collect settings
            self.start_manual()

        if current == 'finished':
            App.get_running_app().stop()


    def do_back(self):

        current = self.root.ids.screen_manager.current

        if current == 'choose_mode':
            self.root.ids.screen_manager.current = 'choose_directories'

            # Disable back button
            self.root.ids.back_button.color = (.9, .9, .9, 1)
            self.root.ids.back_button.diabled = True

            # Enable next button
            self.root.ids.next_button.color = (.23, .23, .23, 1)
            self.root.ids.next_button.diabled = False

        if current == 'choose_methods':
            self.root.ids.screen_manager.current = 'choose_mode'

        if current == 'function_words_choose_features':
            self.root.ids.screen_manager.current = 'choose_methods'

        if current == 'function_words_choose_distance_measure':
            self.root.ids.screen_manager.current = 'function_words_choose_features'

        # N-Grams -------------------------------------------------------------
        if current == 'ngrams_choose_n':
            if self.do_function_words:
                self.root.ids.screen_manager.current = 'function_words_choose_distance_measure'
            else:
                self.root.ids.screen_manager.current = 'choose_methods'

        if current == 'ngrams_choose_most_common_ngrams':
            self.root.ids.screen_manager.current = 'ngrams_choose_n'

        if current == 'ngrams_choose_distance_measure':
            self.root.ids.screen_manager.current = 'ngrams_choose_most_common_ngrams'

        # Burrows' Delta ------------------------------------------------------
        if current == 'delta_choose_most_frequent_words':

            if self.do_ngrams:
                self.root.ids.screen_manager.current = 'ngrams_choose_distance_measure'
            else:
                if self.do_function_words:
                    self.root.ids.screen_manager.current = 'function_words_choose_distance_measure'
                else:
                    self.root.ids.screen_manager.current = 'choose_methods'

        if current == 'delta_choose_culling_factor':
            self.root.ids.screen_manager.current = 'delta_choose_most_frequent_words'

        if current == 'delta_choose_delta_measure':
            self.root.ids.screen_manager.current = 'delta_choose_culling_factor'

    def start(self):

        # Disable back button
        self.root.ids.back_button.color = (.9, .9, .9, 1)
        self.root.ids.back_button.diabled = True

        # Make dir with current date and timestamp
        now = datetime.datetime.now()
        timestamp_dir = self.output_dir / (now.strftime('%Y%m%d_%H%M%S') + '_data')
        os.mkdir(timestamp_dir)

        if self.auto_mode or self.do_function_words:
            self.ana.analyse()
            self.ana.write_results(timestamp_dir)

        if self.auto_mode or self.do_ngrams:
            self.nana.analyse()
            self.nana.write_results(timestamp_dir)

        if self.auto_mode or self.do_delta:
            self.dana.analyse()
            self.dana.write_results(timestamp_dir)

        # Turn next button into exit button
        self.root.ids.next_button.color = (.23, .23, .23, 1)
        self.root.ids.next_button.diabled = False
        self.root.ids.next_button.text = 'Exit'

        self.root.ids.screen_manager.current = 'finished'


    def collect_settings(self):

        # Function words ------------------------------------------------------
        if self.do_function_words:
            self.ana = FeatureAnalyser(self.corpus_reader, self.output_dir, 'manual')

            # Unchoose all features
            for key in self.ana.feature_choice:
                self.ana.feature_choice[key] = False

            if self.root.ids.checkbox_alius.active:
                self.ana.feature_choice['alius'] = True

            if self.root.ids.checkbox_antequam.active:
                self.ana.feature_choice['antequam'] = True

            if self.root.ids.checkbox_conditional.active:
                self.ana.feature_choice['conditional'] = True

            if self.root.ids.checkbox_conjunctions.active:
                self.ana.feature_choice['conjunctions'] = True

            if self.root.ids.checkbox_cum.active:
                self.ana.feature_choice['cum'] = True

            if self.root.ids.checkbox_demonstrative.active:
                self.ana.feature_choice['demonstrative'] = True

            if self.root.ids.checkbox_dum.active:
                self.ana.feature_choice['dum'] = True

            if self.root.ids.checkbox_gerunds.active:
                self.ana.feature_choice['gerunds'] = True

            if self.root.ids.checkbox_idem.active:
                self.ana.feature_choice['idem'] = True

            if self.root.ids.checkbox_indefinite.active:
                self.ana.feature_choice['indefinite'] = True

            if self.root.ids.checkbox_ipse.active:
                self.ana.feature_choice['ipse'] = True

            if self.root.ids.checkbox_iste.active:
                self.ana.feature_choice['iste'] = True

            if self.root.ids.checkbox_personal.active:
                self.ana.feature_choice['personal'] = True

            if self.root.ids.checkbox_priusquam.active:
                self.ana.feature_choice['priusquam'] = True

            if self.root.ids.checkbox_quidam.active:
                self.ana.feature_choice['quidam'] = True

            if self.root.ids.checkbox_quin.active:
                self.ana.feature_choice['quin'] = True

            if self.root.ids.checkbox_reflexive.active:
                self.ana.feature_choice['reflexive'] = True

            if self.root.ids.checkbox_relative.active:
                self.ana.feature_choice['relative'] = True

            if self.root.ids.checkbox_superlatives.active:
                self.ana.feature_choice['superlatives'] = True

            if self.root.ids.checkbox_ut.active:
                self.ana.feature_choice['ut'] = True


            if self.root.ids.function_words_square_euclidean.active:
                self.ana.set_distance_measure('square_euclidean')

            if self.root.ids.function_words_euclidean.active:
                self.ana.set_distance_measure('euclidean')

            if self.root.ids.function_words_manhattan.active:
                self.ana.set_distance_measure('manhattan')

            if self.root.ids.function_words_tanimoto.active:
                self.ana.set_distance_measure('tanimoto')

            if self.root.ids.function_words_matusita.active:
                self.ana.set_distance_measure('matusita')

            if self.root.ids.function_words_clark.active:
                self.ana.set_distance_measure('clark')

            if self.root.ids.function_words_cosine.active:
                self.ana.set_distance_measure('cosine')

            if self.root.ids.function_words_jdivergence.active:
                self.ana.set_distance_measure('jdivergence')

            print()
            print('Settings for function words:')
            print(self.ana.feature_choice)
            print('Distance measure: ' + self.ana.distance_measure[0])

        # Character n-grams -----------------------------------------------
        if self.do_ngrams:

            self.nana = NGramAnalyser(self.corpus_reader, self.output_dir, 'manual')

            if self.root.ids.checkbox_2grams.active:
                self.nana.n = 2

            if self.root.ids.checkbox_3grams.active:
                self.nana.n = 3

            if self.root.ids.checkbox_4grams.active:
                self.nana.n = 4

            if self.root.ids.checkbox_5grams.active:
                self.nana.n = 5

            if self.root.ids.checkbox_6grams.active:
                self.nana.n = 6

            if self.root.ids.checkbox_7grams.active:
                self.nana.n = 7

            if self.root.ids.checkbox_8grams.active:
                self.nana.n = 8

            if self.root.ids.checkbox_9grams.active:
                self.nana.n = 9

            if self.root.ids.checkbox_10grams.active:
                self.nana.n = 10


            if self.root.ids.checkbox_ngrams_50.active:
                self.nana.mf_ngrams = 50

            if self.root.ids.checkbox_ngrams_100.active:
                self.nana.mf_ngrams = 100

            if self.root.ids.checkbox_ngrams_200.active:
                self.nana.mf_ngrams = 200

            if self.root.ids.checkbox_ngrams_300.active:
                self.nana.mf_ngrams = 300

            if self.root.ids.checkbox_ngrams_400.active:
                self.nana.mf_ngrams = 400

            if self.root.ids.checkbox_ngrams_500.active:
                self.nana.mf_ngrams = 500

            if self.root.ids.checkbox_ngrams_1000.active:
                self.nana.mf_ngrams = 1000


            if self.root.ids.ngrams_square_euclidean.active:
                self.nana.set_distance_measure('square_euclidean')

            if self.root.ids.ngrams_euclidean.active:
                self.nana.set_distance_measure('euclidean')

            if self.root.ids.ngrams_manhattan.active:
                self.nana.set_distance_measure('manhattan')

            if self.root.ids.ngrams_tanimoto.active:
                self.nana.set_distance_measure('tanimoto')

            if self.root.ids.ngrams_matusita.active:
                self.nana.set_distance_measure('matusita')

            if self.root.ids.ngrams_clark.active:
                self.nana.set_distance_measure('clark')

            if self.root.ids.ngrams_cosine.active:
                self.nana.set_distance_measure('cosine')

            if self.root.ids.ngrams_jdivergence.active:
                self.nana.set_distance_measure('jdivergence')

            print()
            print('Settings for character n-grams:')
            print(str(self.nana.mf_ngrams)+' most common ' + str(self.nana.n) + '-grams, distance measure: '+self.nana.distance_measure[0])

        # Burrows' Delta ------------------------------------------------------
        if self.do_delta:

            self.dana = DeltaAnalyser(self.corpus_reader, self.output_dir, 'manual')

            if self.root.ids.delta_mfw_50.active:
                self.dana.mfw = 50

            if self.root.ids.delta_mfw_100.active:
                self.dana.mfw = 100

            if self.root.ids.delta_mfw_150.active:
                self.dana.mfw = 150

            if self.root.ids.delta_mfw_200.active:
                self.dana.mfw = 200

            if self.root.ids.delta_mfw_250.active:
                self.dana.mfw = 250

            if self.root.ids.delta_mfw_300.active:
                self.dana.mfw = 300

            if self.root.ids.delta_mfw_400.active:
                self.dana.mfw = 400

            if self.root.ids.delta_mfw_500.active:
                self.dana.mfw = 500

            if self.root.ids.delta_mfw_600.active:
                self.dana.mfw = 600

            if self.root.ids.delta_mfw_700.active:
                self.dana.mfw = 700

            if self.root.ids.delta_mfw_800.active:
                self.dana.mfw = 800

            if self.root.ids.delta_mfw_900.active:
                self.dana.mfw = 900

            if self.root.ids.delta_mfw_1000.active:
                self.dana.mfw = 1000


            if self.root.ids.delta_cull_0.active:
                self.dana.cull = 0

            if self.root.ids.delta_cull_30.active:
                self.dana.cull = 0.3

            if self.root.ids.delta_cull_50.active:
                self.dana.cull = 0.5

            if self.root.ids.delta_cull_70.active:
                self.dana.cull = 0.7

            if self.root.ids.delta_cull_90.active:
                self.dana.cull = 0.9

            if self.root.ids.delta_cull_100.active:
                self.dana.cull = 1


            if self.root.ids.delta_classic.active:
                self.dana.set_delta_measure('delta')

            if self.root.ids.delta_cosine.active:
                self.dana.set_delta_measure('cosine')

            if self.root.ids.delta_eders.active:
                self.dana.set_delta_measure('eders')

            if self.root.ids.delta_argamons.active:
                self.dana.set_delta_measure('argamons')


            print()
            print('Settings for Burrows\' Delta:')
            print('mfw: '+str(self.dana.mfw)+', cull at '+str(self.dana.cull*100)+'%, delta mesaure: '+self.dana.measure[0])

    def start_manual(self):

        print('Start manual.')

        self.root.ids.screen_manager.current = 'progress'

        # Create corpus reader
        self.corpus_reader = CorpusReader(self.corpus_dir)

        self.collect_settings()

        progress_indicator = ProgressIndicator(self.root.ids.task_progress, self.root.ids.overall_progress, self.root.ids.task_label)
        overall_progress_max = 0
        if self.do_function_words:
            self.ana.progress_indicator = progress_indicator
            overall_progress_max += self.ana.progress_max
        if self.do_ngrams:
            self.nana.progress_indicator = progress_indicator
            overall_progress_max += self.nana.progress_max
        if self.do_delta:
            self.dana.progress_indicator = progress_indicator
            overall_progress_max += self.dana.progress_max

        progress_indicator.set_overall_progress_max(overall_progress_max)

        task = threading.Thread(target=self.start)
        task.daemon = True
        task.start()

    def do_auto(self, *args):

        print('Do auto mode.')
        self.auto_mode = True

        # Create corpus reader
        self.corpus_reader = CorpusReader(self.corpus_dir)

        progress_indicator = ProgressIndicator(self.root.ids.task_progress, self.root.ids.overall_progress, self.root.ids.task_label)
        overall_progress_max = 0

        # Function words
        self.ana = FeatureAnalyser(self.corpus_reader, self.output_dir, 'auto')
        self.ana.progress_indicator = progress_indicator
        overall_progress_max += self.ana.progress_max

        # Character n-grams
        self.nana = NGramAnalyser(self.corpus_reader, self.output_dir, 'auto')
        self.nana.progress_indicator = progress_indicator
        overall_progress_max += self.nana.progress_max

        # Burrows' Delta
        self.dana = DeltaAnalyser(self.corpus_reader, self.output_dir, 'auto')
        self.dana.progress_indicator = progress_indicator
        overall_progress_max += self.dana.progress_max

        progress_indicator.set_overall_progress_max(overall_progress_max)

        self.root.ids.screen_manager.current = 'progress'

        task = threading.Thread(target=self.start)
        task.daemon = True
        task.start()

    def do_manual(self, *args):

        print('Do manual mode.')
        self.auto_mode = False

        self.root.ids.screen_manager.current = 'choose_methods'

        self.root.ids.next_button.color = (.23, .23, .23, 1)
        self.root.ids.next_button.diabled = False


def main():
    StyrellaApp().run()


if __name__ == '__main__':
    main()