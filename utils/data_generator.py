""" Module that contains classes and functions to store and generate variables used for the MSPP and MSPP-PD problems formulation"""
from . import random
from . import itertools


class WArc:
    """Class to represent a weighted direct arc

    Attributes:
        i (int): Integer representing the starting node of the arc 
        j (int): Integer representing the ending node of the arc
        w (float): Value to represent the weight of the arc
        idx (int): Unique identifier of the arc within the network
    """

    def __init__(self, begin, end, weight, index):
        """Initialize the instance based on the informations about the arc

        Args:
            i (int): the starting node
            j (int): the ending node
            w (float): the arc's weight
            index (int): its identifier
        """

        self.i = begin
        self.j = end
        self.w = weight
        self.idx = index

    def __repr__(self):
        """Return the representation of the weighted arc instance"""

        return f"{self.__class__.__name__}({self.i!r}, {self.j!r}, {self.w!r}, {self.idx!r})"


class Agent:
    """Class to represent an agent that has to be routed on a network

    Attributes:
        source (int): Integer representing the agent's starting node
        terminus (int): Integer representing the agent's terminus node
        index (int): Unique identifier of the agent within the network
    """

    def __init__(self, source, terminus, index):
        """Initialize the instance based on the informations about the agent

        Args:
            source (int): the source node
            terminus (int): the terminus node
            index (int): its identifier
        """

        self.source = source
        self.terminus = terminus
        self.idx = index
        self.path = None

    def __repr__(self):
        """Return the representation of the agent instance"""

        return f"{self.__class__.__name__}({self.source!r}, {self.terminus!r}, {self.idx!r})"


def network_instances(networks_df):
    """Generator that gives one by one the network instances in the passed dataframe

    Note that a network instance, for us, is completely defined by the set of all its weighted arcs 

    Args:
        networks_df (pd.Dataframe): pandas dataframe containing one or more network instances

    Yields:
        list: a list containing all the weighted arcs of a network instance
    """

    for it in networks_df.index:  # each network instance...
        yield [WArc(i, j, networks_df.loc[it, (i, j)], idx)
               for idx, (i, j) in enumerate(networks_df.columns)]  # ... is formed by a set of arcs


def get_nodes(networks_df):
    """Gives a list with the nodes of the network

    Args:
        networks_df (pd.Dataframe): pandas dataframe containing one or more network instances

    Returns:
        list: a list with all the nodes in the network
    """

    starting_nodes = networks_df.columns.get_level_values(0)
    ending_nodes = networks_df.columns.get_level_values(1)

    return [i for i in starting_nodes.union(ending_nodes).unique()]


def _generate_agents_with_high_simmetry(network_shape, num_of_agents):
    """Generates a number of agents having same source and terminus nodes within the network

    Note that source and terminus nodes are selected randomly from the first and the last columns of
    the network respectively. This function is used inside generate_agents()

    Args:
        network_shape (tuple): (m,n) shape of a grid-like network. Where m is the number of rows
          and n is the number of columns of the network
        num_of_agents (int): number of agents to generate

    Returns:
        list: a list containing the generated agents
    """

    num_of_network_nodes = network_shape[0] * network_shape[1]
    num_of_network_rows = network_shape[0]
    first_network_column = list(range(num_of_network_rows))
    last_network_column = [node + num_of_network_nodes - num_of_network_rows
                           for node in first_network_column]
    agent_idxs = list(range(num_of_agents))
    source_node = random.choice(first_network_column)
    terminus_node = random.choice(last_network_column)

    return [Agent(source_node,
                  terminus_node,
                  idx) for idx in agent_idxs]


def _generate_agents_as_in_paper(network_shape, num_of_agents):
    """Generates a number of agents having source and terminus nodes as described in section 3.3 of the paper

    This function is used inside generate_agents()

    Args:
        network_shape (tuple): (m,n) shape of a grid-like network. Where m is the number of rows
          and n is the number of columns of the network
        num_of_agents (int): number of agents to generate

    Returns:
        list: a list containing the generated agents
    """

    num_of_rows, num_of_cols = network_shape
    num_of_nodes = num_of_rows*num_of_cols
    first_col = list(range(num_of_rows))

    even_agents_sources = first_col[::2]
    odd_agents_sources = first_col[1::2]
    more_agents_sources = itertools.cycle(
        itertools.chain(even_agents_sources, odd_agents_sources))

    agents = []
    for agent_idx in range(num_of_agents):
        if agent_idx < num_of_rows:
            agent_source = agent_idx
        else:
            agent_source = next(more_agents_sources)
        agents.append(Agent(agent_source,
                            agent_source + num_of_nodes - num_of_rows,
                            agent_idx))

    return agents


def _generate_agents_with_low_simmetry(network_shape, num_of_agents):
    """Generates a number of agents having random source and terminus nodes within the network

    Note that source and terminus nodes are selected randomly from the first and the last columns of
    the network respectively. This function is used inside generate_agents()


    Args:
        network_shape (tuple): (m,n) shape of a grid-like network. Where m is the number of rows
          and n is the number of columns of the network
        num_of_agents (int): number of agents to generate

    Returns:
        list: a list containing the generated agents
    """

    num_of_network_nodes = network_shape[0] * network_shape[1]
    num_of_network_rows = network_shape[0]
    first_network_column = list(range(num_of_network_rows))
    last_network_column = [node + num_of_network_nodes - num_of_network_rows
                           for node in first_network_column]

    agent_idxs = list(range(num_of_agents))

    return [Agent(random.choice(first_network_column),
                  random.choice(last_network_column),
                  idx) for idx in agent_idxs]


def generate_agents(network_shape, num_of_agents, *, symmetry="medium"):
    """Generates a number of agents with a specified level of simmetry

    Args:
        network_shape (tuple): (m,n) shape of a grid-like network. Where m is the number of rows
          and n is the number of columns of the network
        num_of_agents (int): number of agents to generate
        symmetry (str): value used to specify the desired level of symmetry that generated agents
          should have (default is "medium")

    Returns:
        list: a list containing the generated agents
    """

    # * Notice that agents are not sorted by idx

    if symmetry == "high":

        return _generate_agents_with_high_simmetry(network_shape, num_of_agents)

    elif symmetry == "medium":

        return _generate_agents_as_in_paper(network_shape, num_of_agents)

    elif symmetry == "low":

        return _generate_agents_with_low_simmetry(network_shape, num_of_agents)
