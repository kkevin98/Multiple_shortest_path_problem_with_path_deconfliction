"""Module that contains functions to plot the solution of a MSPP or a MSPP-PD"""
from . import np, math, plt, cm, ConnectionPatch


def _plot_network_nodes(ax, location_to_node, param_dict):
    """Function to plot the nodes of a network

    Args:
        ax (matplotlib.axes.Axes): Axes object where to plot the nodes
        location_to_node (dict): dictionary that tells which node has to
          be placed in location (x,y) in the axes' plane  
        param_dict (dict): dict containing items to customize the plot
    """

    for loc, node in location_to_node.items():
        ax.plot(*loc, **param_dict)
        ax.annotate(node, loc)


def _plot_network_arcs(ax, w_arcs, node_to_location, param_dict):
    """Function to plot the arcs of a network

    Args:
        ax (matplotlib.axes.Axes): Axes object where to plot the nodes
        w_arcs (list): list of weighted arcs (WArc) in the network
        node_to_location (dict): dictionary that tells on which (x,y) location
          in the axes' plane a node has to be placed 
        param_dict (dict): dict containing items to customize the plot
    """

    for arc in w_arcs:
        arrow = ConnectionPatch(node_to_location[arc.i],
                                node_to_location[arc.j],
                                **param_dict)
        ax.add_artist(arrow)


def _plot_agent_s_t(ax, agent, node_to_location, param_dict):
    """Function to plot the source and terminus of an agent

    Args:
        ax (matplotlib.axes.Axes): Axes object where to plot the 2 nodes
        agent (Agent): agent whose source and terminus we want to print
        node_to_location (dict): dictionary that tells on which (x,y) location
          in the axes' plane a node has to be placed 
        param_dict (dict): dict containing items to customize the plot
    """

    ax.plot(*node_to_location[agent.source], **
            param_dict, label=f"Agent{agent.idx}")
    ax.plot(*node_to_location[agent.terminus], **param_dict)


def _plot_agent_path(ax, x, w_arcs, agent, node_to_location, param_dict):
    """Function to plot the optimal path of an agent

    Args:
        ax (matplotlib.axes.Axes): Axes object where to plot the 2 nodes
        x (np.ndarray): a (2,2) matrix containing the solution of a MSPP or a MSPP-PD
        w_arcs (list): list of weighted arcs (WArc) in the network
        agent (Agent): agent whose path we want to print
        node_to_location (dict): dictionary that tells on which (x,y) location
          in the axes' plane a node has to be placed 
        param_dict (dict): dict containing items to customize the plot
    """

    for arc in w_arcs:
        if math.isclose(x[arc.idx, agent.idx], 1):
            arrow = ConnectionPatch(node_to_location[arc.i],
                                    node_to_location[arc.j],
                                    **param_dict)
            ax.add_artist(arrow)


def plot_solution(x, nodes_grid, w_arcs, agents, title=None):
    """Function to plot the optimal solution of a MSPP or MSPP-PD

    Args:
        x (np.ndarray): a (2,2) matrix containing the solution of a MSPP or a MSPP-PD
        nodes_grid (np.ndarray): a (2,2) matrix containing the nodes of the grid-like
          network
        w_arcs (list): list of weighted arcs (WArc) in the network
        agents (list): list of routed agents (Agent)
        title (str): string representing the title of the plot. By default the plot
          does not have a title
    """

    # create the figure and an iterator over different colors. Each agent will have its own color
    fig, ax = plt.subplots()
    color = iter(cm.Dark2(np.linspace(0, 1, len(agents))))

    # define mapping (x,y)->node and vice versa. Each node has its own position in the plot
    num_rows, num_cols = nodes_grid.shape
    location_to_node = {(x, y): (x+1)*num_rows-y -
                        1 for x in range(num_cols) for y in range(num_rows)}
    node_to_location = {node: pos for pos, node in location_to_node.items()}

    # plot nodes, arcs and agents' paths
    _plot_network_nodes(ax, location_to_node, {"marker": "o", "markersize": 7.5,
                                               "color": "c", "alpha": 0.25})

    _plot_network_arcs(ax, w_arcs, node_to_location, {"coordsA": "data", "coordsB": "data",
                                                      "shrinkA": 5, "shrinkB": 5,
                                                      "arrowstyle": "-|>", "alpha": 0.1})

    for agent in agents:
        agent_color = next(color)
        _plot_agent_s_t(ax, agent, node_to_location,
                        {"marker": "o", "markersize": 7.5, "color": agent_color, "alpha": 0.8})
        _plot_agent_path(ax, x, w_arcs, agent, node_to_location,
                         {"coordsA": "data", "coordsB": "data", "shrinkA": 5, "shrinkB": 5,
                          "arrowstyle": "-|>", "color": agent_color, "linewidth": 2, "alpha": 0.8})

    # plot finalization
    ax.legend(bbox_to_anchor=(1, 1))
    ax.xaxis.set_ticks(range(num_cols))
    ax.yaxis.set_ticks(range(num_rows))
    ax.set_title(title)

    plt.show()
