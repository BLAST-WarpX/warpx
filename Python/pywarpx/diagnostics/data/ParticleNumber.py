from BaseReader import DataReader


class ParticleNumberData(DataReader):
    """
    Reader for the ParticleNumber reduced diagnostic.
    """

    def __init__(self, red_diags_dir="./diags/reducedfiles", separator=" ", fname='ParticleNumber.txt'):
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

        super().__init__(red_diags_dir, separator)

        self.fname = fname
        if self.fname is None:
            raise ValueError('The diagnostic file name cannot be None!')

        self.has_species = True

    def get_nspecies(self):
        # remove 4 columns: step, time, total_macroparticles, total_weight
        # divide by 2: we save both the total number of macroparticles
        # and the total number of real particles for every species
        return DataReader.get_nspecies(self, columns_speciesless=4., columns_per_species=2.)

    def get_species_names(self):
        # remove first 3 entries (step, time, total) then select every 1 entry
        return DataReader.get_species_names(self, string='_macroparticles\(\)', initial_columns=3, columns_per_species=1)

    def get_valid_args(self):
        """
        Returns the valid strings to extract the data.
        """

        species_names = self.get_species_names()
        valid_args = ('total_macroparticles', 'total_weight',
                      *(s+'_macroparticles' for s in species_names), *(s+'_weight' for s in species_names))
        return valid_args

    def get_data(self, *args, steps=None, times=None):
        """
        Arguments:
            string:
                Can be one or more among the valid arguments.
        Keyword arguments:
            steps = list or np.array of integers or None (optional):
                Timesteps at which the desidered output will be returned.
                If equal to None or not specified then all timesteps are given.
            times = list or np.array of numbers or None (optional):
                The desidered output will be returned at the closest availble times.
                If equal to None or not specified then all timesteps are given
        Output:
            pandas dataframe with columns: steps, times, requested data
        """

        valid_args = self.get_valid_args()

        # get data using parent class
        data = DataReader.get_data(self, valid_args, steps, times, *args)

        # restrict to the desidered steps or times
        restricted_data = self.restrict_data(data, steps, times)

        # rename columns
        restricted_data.columns = ['steps', 'times', *[name for name in args]]

        return restricted_data
