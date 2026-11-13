
from BaseReader import DataReader


class ParticleMomentumData(DataReader):
    """
    Reader for the ParticleMomentum reduced diagnostic.
    """

    def __init__(self, red_diags_dir="./diags/reducedfiles", separator=" ", fname='ParticleMomentum.txt'):
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
        # remove 8 columns: step, time, total_x, total_y, total_z, total_mean_x, total_mean_y, total_mean_z
        # divide by 6: for each species we save both the total and mean momentum along the 3 coordinates
        return DataReader.get_nspecies(self, columns_speciesless=8., columns_per_species=2.*3.)

    def get_species_names(self):
        return DataReader.get_species_names(self, string='_x\(kg\*m\/s\)', initial_columns=5, columns_per_species=3)

    def get_valid_args(self):
        """
        Returns the valid strings to extract the data.
        """

        species_names = self.get_species_names()
        valid_args = ('total_x', 'total_y', 'total_z',
                      'total_mean_x', 'total_mean_y', 'total_mean_z',
                      *(s+'_x' for s in species_names), *(s+'_y' for s in species_names), *(s+'_z' for s in species_names),
                      *(s+'_mean_x' for s in species_names), *(s+'_mean_y' for s in species_names), *(s+'_mean_z' for s in species_names))
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
