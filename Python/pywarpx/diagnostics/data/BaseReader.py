import collections
import os
import re

import numpy as np
import pandas as pd


class DataReader():
    """
    Base class that all data readers should inherit from.
    Works only in Cartesian geometry.
    """

    def __init__(self, red_diags_dir="./diags/reducedfiles", separator=" ", fname=None):
        """
        Parameters
        ----------
        red_diags_dir: string
            Path to the reduced diagnostic directory, default is './diags/reducedfiles'.
        fname: string
            Name of the reduced diagnostic file.
        separator: string
            The separator between row values in the output file, default is a whitespace " ".
        """

        if red_diags_dir is None:
            raise ValueError('The red_diags_dir parameter cannot be None!')

        self.red_diags_dir = red_diags_dir
        self.separator = separator
        self.fname = None

        self.has_species = None

    def get_data_path(self):
        """
        Returns
        -------
        A string with the path to the current diagnostic (assumed unique).
        """

        if not os.path.isdir(self.red_diags_dir):
            raise IOError(f'The reduced diagnostic directory {self.red_diags_dir} does not exist.\n'
                          'Did you set the proper path to the reduced diagnostic directory?\n'
                          'Did the simulation already run?')

        data_file_path = os.path.join(
            self.red_diags_dir,
            self.fname
        )

        if not os.path.isfile(data_file_path):
            raise IOError(f'The file {data_file_path} does not exist.\n'
                          'Did you type the correct filename?')

        return data_file_path

    def get_ncols(self):
        ''' Returns the number of columns of the reduced diagnostic.'''
        data_file_path = self.get_data_path()
        header=pd.read_csv(data_file_path, header=0, sep=self.separator, nrows=0)
        ncols = len(header.columns)
        return ncols

    def get_nspecies(self, columns_speciesless=None, columns_per_species=None):
        '''Returns the number of species involved in the diagnostic.'''
        # deduces number of species from number of columns
        if self.has_species == True:
            ncols = self.get_ncols()
            # remove <columns_speciesless> columns and divide to avoid counting multiple entries
            nspecies = int((ncols - columns_speciesless)/columns_per_species)
            return nspecies
        else:
            return 0

    def get_species_names(self, string=None, initial_columns=None, columns_per_species=None):
        '''Returns the names of the species involved in the diagnostic.'''
        # deduces names of species from the header by considering only the relevant entries
        # finds the string by locating the units
        if self.has_species == True:
            species_names = []
            data_file_path = self.get_data_path()
            nspecies = self.get_nspecies()

            header = pd.read_csv(data_file_path, header=0, sep=self.separator, nrows=0)
            colnames = header.columns.values
            for i in range(nspecies):
                result=re.search('](.+?)'+string,colnames[initial_columns+i*columns_per_species])
                species_names.append(result.group(1))
            return species_names
        else:
            raise ValueError('The current diagnostics does not involve any species!')

    def load_data(self):
        data_file_path = self.get_data_path()
        df = pd.read_csv(data_file_path, sep=self.separator, header=0)
        return df

    def get_steps(self):
        data_file_path = self.get_data_path()
        df = pd.read_csv(data_file_path, sep=self.separator, usecols=['#[0]step()'])
        steps = np.unique(df.values)
        return steps

    def get_times(self):
        data_file_path = self.get_data_path()
        df = pd.read_csv(data_file_path, sep=self.separator, usecols=['[1]time(s)'])
        times = np.unique(df.values)
        return times

    def get_data(self, valid_args, steps=None, times=None, *args):

        df = self.load_data()
        all_cols = df.columns

        col = []
        data = []

        for a in args:
            if a not in valid_args:
                raise ValueError(f'{a} is an invalid argument!\n'
                                  'list of valid arguments: \n'
                                  '{valid_args}')
            else:
                col.append([c for c in all_cols if a in c][0])

        if col == []:
            raise ValueError('Could not find any valid column names!\n'
                             'Is the spelling correct?\n'
                             'Did you select any valid data?')

        col = ['#[0]step()','[1]time(s)'] + col
        data = df[col]

        return data


    def restrict_data(self, data, steps=None, times=None): # check if works for all diags (e.g. probes, field reduction)
        """
        Restricts the data to the requested steps or times.
        Arguments:
            pandas dataframe:
                Extracted data.
        Keyword arguments:
            steps = list or np.array of integers or None (optional):
                Timesteps at which the desidered output will be returned.
                If equal to None or not specified then all timesteps are given.
            times = list or np.array of numbers or None (optional):
                The desidered output will be returned at the closest available time.
                If equal to None or not specified then all timesteps are given.
        Output:
            pandas dataframe with columns: steps, times, extracted data
        """

        all_times = self.get_times()
        all_steps = self.get_steps()

        if (steps is not None) and (times is not None):
            raise ValueError('Please provide only one between steps and times!')
        elif steps is not None:
            # make steps iterable if they are not
            if not isinstance(steps, collections.abc.Iterable):
                steps = np.array([steps])
        elif times is not None:
            if not isinstance(times, collections.abc.Iterable):
                times = np.array([times])
            steps = []
            for t in times:
                i = np.argmin((t-all_times)**2)
                steps = np.append(steps, all_steps[i])
        else:
            steps = all_steps

        # verify that requested iterations exist
        if not set(steps).issubset(all_steps):
            raise IndexError(f'Selected step {steps} is not available!\n'
                             'List of available steps: \n'
                             '{all_steps}')

        data = data.loc[data['#[0]step()'].isin(steps)]
        # reset index
        data = data.reset_index(drop=True)

        return data
