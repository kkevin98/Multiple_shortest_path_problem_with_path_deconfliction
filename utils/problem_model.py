from . import gb
from . import GRB


class WArc:

    def __init__(self, begin, end, weight, index):
        self.i = begin
        self.j = end
        self.w = weight
        self.idx = index

    def __repr__(self):
        return f"{self.__class__.__name__}({self.i!r}, {self.j!r}, {self.w!r}, {self.idx!r})"


class Agent:

    def __init__(self, source, terminus, index):
        # ? Consider agent name
        self.source = source
        self.terminus = terminus
        self.idx = index
        self.path = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.source!r}, {self.terminus!r}, {self.idx!r})"


def get_nodes(networks_df):
    '''Function that gives a list with all the nodes in the network'''

    starting_nodes = networks_df.columns.get_level_values(0)
    ending_nodes = networks_df.columns.get_level_values(1)

    return [i for i in starting_nodes.union(ending_nodes).unique()]


def network_instances(networks_df):
    '''Functions that gives one by one the network instaces of the dataframe in the form of a list of arcs '''

    for it in networks_df.index:  # each network instance...
        yield [WArc(i, j, networks_df.loc[it, (i, j)], idx)
               for idx, (i, j) in enumerate(networks_df.columns)]  # ... is formed by a set of arcs


def _agent_should_start_on_node(agent_idx, num_of_network_rows, node):
    '''Tells if an agent with a certain index ('agent_idx') should start on 'node' or not'''

    return agent_idx % num_of_network_rows == node


def generate_agents(network_shape, congestion_level):
    '''Generates a list of agents from a specified congestion level of a network of shape network_shape'''

    # * Notice that agents are not sorted by idx

    num_of_network_rows = network_shape[0]
    num_of_agents = int(num_of_network_rows * congestion_level)
    num_of_nodes = num_of_network_rows * network_shape[1]
    first_network_column = list(range(num_of_network_rows))
    agent_idxs = list(range(num_of_agents))

    return [Agent(node,
                  node + num_of_nodes - num_of_network_rows,
                  agent_idx)
            for node in first_network_column
            for agent_idx in agent_idxs if _agent_should_start_on_node(agent_idx, num_of_network_rows, node)]


def _compute_flow(X, node, w_arcs, agent):
    flow_out = gb.quicksum(
        X[arc.idx, agent.idx]
        for arc in w_arcs if arc.i == node
    )
    flow_in = gb.quicksum(
        X[arc.idx, agent.idx]
        for arc in w_arcs if arc.j == node
    )
    return flow_out - flow_in


def set_MSPP(nodes, w_arcs, agents):

    MSPP_pb = gb.Model()
    MSPP_pb.setParam("OutputFlag", 0)

    # Decision variables
    X_var_shape = len(w_arcs), len(agents)
    X = MSPP_pb.addMVar(X_var_shape,
                        vtype=GRB.BINARY,  # 5) Binary constraints
                        name="X")

    # 1-3) Objective
    distance_obj = gb.quicksum(
        arc.w * X[arc.idx, agent.idx]
        for arc in w_arcs for agent in agents
    )
    MSPP_pb.setObjectiveN(distance_obj, index=0, weight=1, name="Distance")

    # 4) Flow constraints
    for agent in agents:
        for node in nodes:
            if node == agent.source:
                MSPP_pb.addConstr(_compute_flow(X, node, w_arcs, agent) == 1)
            elif node == agent.terminus:
                MSPP_pb.addConstr(_compute_flow(X, node, w_arcs, agent) == -1)
            else:
                MSPP_pb.addConstr(_compute_flow(X, node, w_arcs, agent) == 0)

    return MSPP_pb, X


def set_ABP(nodes, w_arcs, agents):

    MSPP_PD_ABP_pb, X = set_MSPP(nodes, w_arcs, agents)

    # Additional decision variables
    Psi_var_shape = len(w_arcs)
    Psi = MSPP_PD_ABP_pb.addMVar(Psi_var_shape,
                                 vtype=GRB.BINARY,  # 8) Binary constraints
                                 name="Psi")

    # 6) Additional objective
    penalty_obj = gb.quicksum(
        Psi[arc.idx] for arc in w_arcs
    )
    MSPP_PD_ABP_pb.setObjectiveN(
        penalty_obj, index=1, weight=1, name="Penalty")

    # 7) Additonal constraints
    for arc in w_arcs:
        MSPP_PD_ABP_pb.addConstr(
            1/len(agents) *
            (gb.quicksum(X[arc.idx, agent.idx] for agent in agents) - 1)
            <= Psi[arc.idx]
        )

    return MSPP_PD_ABP_pb, X, Psi


def set_NBP(nodes, w_arcs, agents):

    MSPP_PD_NBP_pb, X = set_MSPP(nodes, w_arcs, agents)

    # Additional decision variables
    R_var_shape = len(nodes), len(agents)
    R = MSPP_PD_NBP_pb.addMVar(R_var_shape,
                               vtype=GRB.BINARY,  # 13) Binary constraints
                               name="R")
    Xi_var_shape = len(nodes)
    Xi = MSPP_PD_NBP_pb.addMVar(Xi_var_shape,
                                vtype=GRB.BINARY,  # 14) Binary constraints
                                name="Xi")

    # 9) Additional objective
    penalty_obj = gb.quicksum(
        Xi[node] for node in nodes
    )
    MSPP_PD_NBP_pb.setObjectiveN(
        penalty_obj, index=1, weight=1, name="Penalty")

    # 10,11) Turning on r_i constraints
    for agent in agents:
        for node in nodes:
            MSPP_PD_NBP_pb.addConstr(
                R[node, agent.idx] >= gb.quicksum(X[arc.idx, agent.idx]
                                                  for arc in w_arcs if arc.i == node)
            )
            MSPP_PD_NBP_pb.addConstr(
                R[node, agent.idx] >= gb.quicksum(X[arc.idx, agent.idx]
                                                  for arc in w_arcs if arc.j == node)
            )

    # 12) Turning on xi_i constraints
    #! Seems weird, the -1 should be outside the summation
    for node in nodes:
        MSPP_PD_NBP_pb.addConstr(
            1/len(agents) * (gb.quicksum(R[node, agent.idx] - 1
                                         for agent in agents))
        )

    return MSPP_PD_NBP_pb, X, R, Xi


def set_problem(problem_type, nodes, w_arcs, agents):
    params = nodes, w_arcs, agents
    if problem_type == "MSPP":
        return set_MSPP(*params)
    elif problem_type == "ABP":
        return set_ABP(*params)
    elif problem_type == "NBP":
        return set_NBP(*params)
