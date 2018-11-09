__description__ = \
"""
A class for running abitrarily complex first-order chemical kinetics markov
models based on simple, human-readable string-basd input.
"""
__author__ = "Michael J. Harms (harmsm@gmail.com)"
__date__ = "2018-11-07"

import re, copy
import numpy as np

from matplotlib import pyplot as plt

class ReactionSimulator:
    """
    Class for holding and manipulating a reaction scheme.
    """

    def __init__(self,rxn_input=None):
        """
        Initialize reaction.  If rxn_input is specified, load the reaction
        in.  If not, initialize an empty class.

        rxn_input: None, string, or text file specifying the reaction (see
                   load_rxn docstring for format)
        """

        self._rxn_loaded = False

        self._input_string = None
        if rxn_input is not None:
            self.load_rxn(rxn_input)

    def load_rxn(self,rxn_input,min_self_prob=0.9):
        """
        Load a reaction in from a string or text file.

        rxn_input: string or text file describing reaction (see below)
        min_self_prob: float, 0 < min_self_prob < 1.  Indicates the self
                       probability for the species with the largest magnitude
                       set of transitions away from itself.  The larger the
                       value, the smaller the time step.

        An example reaction is below:

        A = 10
        B = 0
        A -> B 5

        The first two lines are initial concentrations of A and B, the last
        line is the reaction (A -> B with rate 5).  To make reversible, add
        something like "B -> A 1" to give a back reaction with rate 1.  You
        can specify as many reactions as you want, with as many species as you
        want.  You must specify an initial concentration for any species
        that is involved in any reaction.

        Lines starting with "#" are ignored as comments.
        """

        if min_self_prob <= 0 or min_self_prob >= 1:
            err = "min_self_prob must be between 0 and 1 (not-inclusive)\n"
            raise ValueError(err)

        self._min_self_prob = min_self_prob

        try:
            f = open(rxn_input,"r")
            self._input_string = f.read()
            f.close()
        except FileNotFoundError:
            self._input_string = rxn_input

        self._parse_reaction()
        self._construct_matrix()

        self._rxn_loaded = True


    def take_step(self,num_steps=1):
        """
        Advance the model by num_steps time steps.
        """

        if num_steps > 1:
            T = np.linalg.matrix_power(self.T,num_steps)
        else:
            T = self.T

        self._current_conc = np.dot(T,self._current_conc)


        self._time_steps.append(self._time_steps[-1]+num_steps)
        self._conc_history.append(self._current_conc)


    def _parse_reaction(self):
        """
        Parse a string describing a set of reactions.
        """

        # Split on newlines
        lines = self._input_string.split("\n")

        # Use this pattern to look for reaction lines
        rxn_pattern = re.compile("->")

        # Data structures to populate
        species_seen = []
        reactions = {}
        species_list = []
        conc_list = []

        # Go through every line
        for line in lines:

            # skip blank lines and comments
            if line.strip() == "" or line.startswith("#"):
                continue

            # reaction line
            if rxn_pattern.search(line):

                # Split on ->; should yield exactly two fields
                rxn = line.split("->")
                if len(rxn) != 2:
                    err = "mangled reaction line\n ({})\n".format(line)
                    raise ValueError(err)

                # reactant is first field
                reactant = rxn[0].strip()

                # split second field; should have exactly two outputs
                product_and_rate = rxn[1].split()
                if len(product_and_rate) != 2:
                    err = "mangled reaction line\n ({})\n".format(line)
                    raise ValueError(err)

                # product is first output
                product = product_and_rate[0].strip()

                # rate is second output
                try:
                    rate = float(product_and_rate[1])
                except ValueError:
                    err = "mangled reaction line (rate not a float)\n ({})\n".format(line)
                    raise ValueError(err)

                # Reaction key defines what reaction is specified
                reaction_key = (reactant,product)
                try:

                    # Make sure this reaction has not been seen before
                    reactions[reaction_key]
                    err = "reaction defined more than once ({})\n".format(reaction_key)
                    raise ValueErro(err)

                # Record reaction, rate, and what species have been seen
                except KeyError:
                    reactions[reaction_key] = rate

                    species_seen.append(reactant)
                    species_seen.append(product)

            else:

                # Assume this is a concentration line.  split on =.  Must have two fields
                conc_line = line.split("=")
                if len(conc_line) != 2:
                    err = "mangled concentration line\n ({})\n".format(line)
                    raise ValueError(err)

                # First field is species name.  Check to see if it has been seen before.
                species = conc_line[0].strip()
                if species in species_list:
                    err = "duplicate species concentration ({})\n".format(species)
                    raise ValueError(err)

                # second field is concentration. must be float.
                try:
                    conc = float(conc_line[1])
                except ValueError:
                    err = "mangled concentration line\n ({})\n".format(line)
                    raise ValueError(err)

                # Record the species and concentration in lists (ordered!)
                species_list.append(species)
                conc_list.append(conc)

        # Unique set of species observed in reactions
        species_seen = set(species_seen)

        # Make sure that there is a concentration specified for every species in
        # a reaction.
        if not species_seen.issubset(set(species_list)):
            err = "not all species have initial concentrations\n"
            raise ValueError(err)

        # Final lists of species names
        self._species_list = species_list[:]
        self._conc_list = conc_list[:]
        self._reactions = reactions

        # Concentrations as arrays
        self._initial_conc = np.array(self._conc_list)
        self._current_conc = np.array(self._conc_list)

        # Keep track of changes
        self._time_steps = [0]
        self._conc_history = [self._current_conc]

        # Number of species
        self._n = len(self._species_list)

    def _construct_matrix(self):
        """
        Construct a transition matrix given the set of reactions.
        """

        # Make sure a reaction has been loaded in
        try:
            self._species_list
        except AttributeError:
            err = "matrix cannot be constructed until reaction is loaded\n"
            raise ValueError(err)

        # construct the rate matrix
        self._rate_matrix = np.zeros((self._n,self._n),dtype=float)

        # Go through every reactant/product pair
        for i, reactant in enumerate(self._species_list):
            for j, product in enumerate(self._species_list):

                # skip self reaction
                if reactant == product:
                    continue

                # Look for reaction between these species. If specified, add a rate.
                try:
                    rate = self._reactions[(reactant,product)]
                    self._rate_matrix[j,i] = rate
                except KeyError:
                    pass

        # Figure out dt.  This is tuned so that the species with the highest
        # total out rate has a self probability of self._min_self_prob.
        highest_total_out_rate = np.max(np.sum(self._rate_matrix,0))
        self._dt = (1 - self._min_self_prob)/highest_total_out_rate

        # construct transition matrix, adding self-probabilities
        self._T = np.copy(self._rate_matrix)*self._dt
        p_self = (1 - np.sum(self._T,0))*np.eye(self._n)
        self._T = self._T + p_self


    @property
    def species(self):
        """
        List of species names.
        """

        if not self._rxn_loaded:
            return None

        return self._species_list

    @property
    def reactions(self):
        """
        Dictionary of reactions, with keys like (reactant,product) and values
        of rates.
        """

        if not self._rxn_loaded:
            return None

        return self._reactions

    @property
    def dt(self):
        """
        Time step.
        """

        if not self._rxn_loaded:
            return None

        return self._dt

    @property
    def T(self):
        """
        Transition matrix.
        """

        if not self._rxn_loaded:
            return None

        return self._T

    @property
    def initial_conc(self):
        """
        Initial concentrations of all species.
        """

        if not self._rxn_loaded:
            return None

        return self._initial_conc

    @property
    def conc(self):
        """
        Current concentrations of all species.
        """

        if not self._rxn_loaded:
            return None

        return self._current_conc

    @property
    def t(self):
        """
        Time steps sampled.
        """

        return np.array(self._time_steps)*self._dt

    @property
    def conc_history(self):
        """
        Concentrations over steps.
        """

        return np.array(self._conc_history)

def run_and_plot(r,species=None,num_steps=100,step_size=1):

    x = copy.deepcopy(r)

    if species is None:
        species_list = [i for i in range(len(x.species))]
    else:
        try:
            species_list = [x.species.index(species)]
        except ValueError:
            err = "species {} not found in reaction scheme. species available:\n\n"
            err += ",".join(x.species)
            raise ValueError(err)

    for i in range(num_steps):
        x.take_step(step_size)

    for i in species_list:
        plt.plot(x.t,x.conc_history[:,i])

    plt.xlabel("time (s)")
    plt.ylabel("[species]")
    #plt.legend()
