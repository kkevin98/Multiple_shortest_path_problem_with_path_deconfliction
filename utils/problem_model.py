"""Module that contains useful functions to create or deal with MSPPs and MSPP-PDs"""
from . import gb
from . import GRB


def _compute_flow(X, node, w_arcs, agent):
    """Compute the flow in a given node for a particular agent

    The flow is defined as the difference between outgoing and ingoing edges traversed by the agent in that node

    Args:
        X (gb.MVar): X decision variables of a MSPP or MSPP-PD
        node (int): node under consideration
        w_arcs (list): list of weighted arcs that represent the network instance
        agent (Agent): the agent for which to compue the flow

    Returns:
        gb.MLinExpr: an expression for the computed flow 
    """

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
    """Create and set a MSPP for a network instance given the agents to route

    Args:
        nodes (list): list of the nodes in the network instance
        w_arcs (list): list of weighted arcs in the network instance
        agents (list): list of agents that has to be routed

    Returns:
        tuple: a tuple (MSPP_pb, X) where:
          - MSPP_pb is a gb.Model that represent the created MSPP
          - X is a gb.MVar containing the decision variables associated to the agents' paths
    """

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
    """Create and set a MSPP-PD(ABP) for a network instance given the agents to route

    Args:
        nodes (list): list of the nodes in the network instance
        w_arcs (list): list of weighted arcs in the network instance
        agents (list): list of agents that has to be routed

    Returns:
        tuple: a tuple (MSPP_PD_ABP_pb, X, Psi) where:
          - MSPP_PD_ABP_pb is a gb.Model that represent the created MSPP-PD(ABP)
          - X is a gb.MVar containing the decision variables associated to the agents' paths
          - Psi is a gb.MVar containing the decison variables that tell if more than one agents
            traverse a specific arc
    """

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
    """Create and set a MSPP-PD(NBP) for a network instance given the agents to route

    Args:
        nodes (list): list of the nodes in the network instance
        w_arcs (list): list of weighted arcs in the network instance
        agents (list): list of agents that has to be routed

    Returns:
        tuple: a tuple (MSPP_PD_NBP_pb, X, R, Xi) where:
          - MSPP_PD_NBP_pb is a gb.Model that represent the created MSPP-PD(NBP)
          - X is a gb.MVar containing the decision variables associated to the agents' paths
          - R is a gb.MVar containing the decision variables that tell if a particular
            agent traverse a particular node
          - Zeta is a gb.MVar containing the decision variables that tell if more than one agent
            traverse a particular node
    """

    MSPP_PD_NBP_pb, X = set_MSPP(nodes, w_arcs, agents)

    # Additional decision variables
    R_var_shape = len(nodes), len(agents)
    R = MSPP_PD_NBP_pb.addMVar(R_var_shape,
                               vtype=GRB.BINARY,  # 13) Binary constraints
                               name="R")
    Zeta_var_shape = len(nodes)
    Zeta = MSPP_PD_NBP_pb.addMVar(Zeta_var_shape,
                                  vtype=GRB.BINARY,  # 14) Binary constraints
                                  name="Zeta")

    # 9) Additional objective
    penalty_obj = gb.quicksum(
        Zeta[node] for node in nodes
    )
    MSPP_PD_NBP_pb.setObjectiveN(
        penalty_obj, index=1, weight=1, name="Penalty")

    # 10,11) Turning on r_i constraints
    for arc in w_arcs:
        for agent in agents:
            MSPP_PD_NBP_pb.addConstr(
                R[arc.i, agent.idx] >= X[arc.idx, agent.idx]
            )
            MSPP_PD_NBP_pb.addConstr(
                R[arc.j, agent.idx] >= X[arc.idx, agent.idx]
            )

    # 12) Turning on xi_i constraints
    # ! Different from paper. Paper seems weird, the -1 should be outside the summation
    for node in nodes:
        MSPP_PD_NBP_pb.addConstr(
            1/len(agents) *
            (gb.quicksum(R[node, agent.idx] for agent in agents) - 1)
            <= Zeta[node]
        )

    return MSPP_PD_NBP_pb, X, R, Zeta


def set_ALP(nodes, w_arcs, agents):
    """Create and set a MSPP-PD(ALP) for a network instance given the agents to route

    Args:
        nodes (list): list of the nodes in the network instance
        w_arcs (list): list of weighted arcs in the network instance
        agents (list): list of agents that has to be routed

    Returns:
        tuple: a tuple (MSPP_PD_ALP_pb, X, Eps) where:
          - MSPP_PD_ALP_pb is a gb.Model that represent the created MSPP-PD(ALP)
          - X is a gb.MVar containing the decision variables associated to the agents' paths
          - Eps is a gb.MVar containing the decision variables that tell if a any agent
            traverse a particular arc
    """

    MSPP_PD_ALP_pb, X = set_MSPP(nodes, w_arcs, agents)

    # Additional decision variables
    Eps_var_shape = len(w_arcs)
    Eps = MSPP_PD_ALP_pb.addMVar(Eps_var_shape,
                                 vtype=GRB.BINARY,  # 17) Binary constraints
                                 name="Eps")

    # 15) Additional objective
    penalty_obj = gb.quicksum(
        - Eps[arc.idx] + gb.quicksum(X[arc.idx, agent.idx] for agent in agents)
        for arc in w_arcs
    )
    MSPP_PD_ALP_pb.setObjectiveN(
        penalty_obj, index=1, weight=1, name="Penalty"
    )

    # 16) Turning on eps_i constraints
    for arc in w_arcs:
        MSPP_PD_ALP_pb.addConstr(
            1/len(agents) *
            (gb.quicksum(X[arc.idx, agent.idx] for agent in agents))
            <= Eps[arc.idx]
        )
        MSPP_PD_ALP_pb.addConstr(
            Eps[arc.idx] <= gb.quicksum(X[arc.idx, agent.idx]
                                        for agent in agents)
        )

    return MSPP_PD_ALP_pb, X, Eps


def set_NLP(nodes, w_arcs, agents):
    """Create and set a MSPP-PD(NLP) for a network instance given the agents to route

    Args:
        nodes (list): list of the nodes in the network instance
        w_arcs (list): list of weighted arcs in the network instance
        agents (list): list of agents that has to be routed

    Returns:
        tuple: a tuple (MSPP_PD_NLP_pb, X, R, Theta) where:
          - MSPP_PD_NLP_pb is a gb.Model that represent the created MSPP-PD(NLP)
          - X is a gb.MVar containing the decision variables associated to the agents' paths
          - R is a gb.MVar containing the decision variables that tell if a particular
            agent traverse a particular node
          - Theta is a gb.MVar containing the decision variables that tell if any agent
            traverse a particular node
    """

    MSPP_PD_NLP_pb, X = set_MSPP(nodes, w_arcs, agents)

    # Additional decision variables
    R_var_shape = len(nodes), len(agents)
    R = MSPP_PD_NLP_pb.addMVar(R_var_shape,
                               vtype=GRB.BINARY,  # 13) Binary constraints
                               name="R")
    Theta_var_shape = len(nodes)
    Theta = MSPP_PD_NLP_pb.addMVar(Theta_var_shape,
                                   vtype=GRB.BINARY,  # 20) Binary constraints
                                   name="Theta")

    # 18) Additional objective
    penalty_obj = gb.quicksum(
        - Theta[node] + gb.quicksum(R[node, agent.idx] for agent in agents)
        for node in nodes
    )
    MSPP_PD_NLP_pb.setObjectiveN(
        penalty_obj, index=1, weight=1, name="Penalty"
    )

    # 10,11) Turning on r_i constraints
    for arc in w_arcs:
        for agent in agents:
            MSPP_PD_NLP_pb.addConstr(
                R[arc.i, agent.idx] >= X[arc.idx, agent.idx]
            )
            MSPP_PD_NLP_pb.addConstr(
                R[arc.j, agent.idx] >= X[arc.idx, agent.idx]
            )

    # 19) Turning on theta_i constraints
    for node in nodes:
        MSPP_PD_NLP_pb.addConstr(
            1/len(agents) *
            (gb.quicksum(R[node, agent.idx] for agent in agents))
            <= Theta[node]
        )
        MSPP_PD_NLP_pb.addConstr(
            Theta[node] <= gb.quicksum(R[node, agent.idx]
                                       for agent in agents)
        )

    return MSPP_PD_NLP_pb, X, R, Theta


def set_AQP(nodes, w_arcs, agents):
    """Create and set a MSPP-PD(AQP) for a network instance given the agents to route

    Args:
        nodes (list): list of the nodes in the network instance
        w_arcs (list): list of weighted arcs in the network instance
        agents (list): list of agents that has to be routed

    Returns:
        tuple: a tuple (MSPP_PD_AQP_pb, Z) where:
          - MSPP_PD_AQP_pb is a gb.Model that represent the created MSPP-PD(AQP)
          - X is a gb.MVar containing the decision variables associated to the agents' paths
          - Z is a gb.MVar containing the decision variables used to linearize the original
            objective function of the MSPP-PD(AQP)
    """

    MSPP_PD_AQP_pb, X = set_MSPP(nodes, w_arcs, agents)

    # Additional decision variables
    Z_var_shape = len(w_arcs), len(agents), len(agents)
    Z = MSPP_PD_AQP_pb.addMVar(Z_var_shape,
                               vtype=GRB.BINARY,  # 26) Binary constraints
                               name="Z")

    # 22) Additional (linearized) objective
    penalty_obj = gb.quicksum(
        Z[arc.idx, agent.idx, agent_.idx] for arc in w_arcs
        for agent in agents for agent_ in agents if agent_.idx < agent.idx
    )
    MSPP_PD_AQP_pb.setObjectiveN(
        penalty_obj, index=1, weight=1, name="Penalty"
    )

    # 23-25) Well-defined Z variable
    for arc in w_arcs:
        for agent in agents:
            for agent_ in agents:
                if agent_.idx < agent.idx:
                    MSPP_PD_AQP_pb.addConstr(
                        Z[arc.idx, agent.idx, agent_.idx] <= X[arc.idx, agent.idx]
                    )
                    MSPP_PD_AQP_pb.addConstr(
                        Z[arc.idx, agent.idx, agent_.idx] <= X[arc.idx, agent_.idx]
                    )
                    MSPP_PD_AQP_pb.addConstr(
                        Z[arc.idx, agent.idx, agent_.idx] >=
                        X[arc.idx, agent.idx] + X[arc.idx, agent_.idx] - 1
                    )

    return MSPP_PD_AQP_pb, X, Z


def set_NQP(nodes, w_arcs, agents):
    """Create and set a MSPP-PD(NQP) for a network instance given the agents to route

    Args:
        nodes (list): list of the nodes in the network instance
        w_arcs (list): list of weighted arcs in the network instance
        agents (list): list of agents that has to be routed

    Returns:
        tuple: a tuple (MSPP_PD_NQP_pb, R, W) where:
          - MSPP_PD_AQP_pb is a gb.Model that represent the created MSPP-PD(AQP)
          - X is a gb.MVar containing the decision variables associated to the agents' paths
          - R is a gb.MVar containing the decision variables that tell if a particular
            agent traverse a particular node
          - W is a gb.MVar containing the decision variables used to linearize the original
            objective function of the MSPP-PD(AQP)
    """

    MSPP_PD_NQP_pb, X = set_MSPP(nodes, w_arcs, agents)

    # Additional decision variables
    R_var_shape = len(nodes), len(agents)
    R = MSPP_PD_NQP_pb.addMVar(R_var_shape,
                               vtype=GRB.BINARY,  # 13) Binary constraints
                               name="R")
    W_var_shape = len(nodes), len(agents), len(agents)
    W = MSPP_PD_NQP_pb.addMVar(W_var_shape,
                               vtype=GRB.BINARY,  # 33) Binary constraints
                               name="W")

    # 28) Additional (linearized) objective
    penalty_obj = gb.quicksum(
        W[node, agent.idx, agent_.idx] for node in nodes
        for agent in agents for agent_ in agents if agent_.idx < agent.idx
    )
    MSPP_PD_NQP_pb.setObjectiveN(
        penalty_obj, index=1, weight=1, name="Penalty"
    )

    # 10,11) Turning on r_i constraints
    for arc in w_arcs:
        for agent in agents:
            MSPP_PD_NQP_pb.addConstr(
                R[arc.i, agent.idx] >= X[arc.idx, agent.idx]
            )
            MSPP_PD_NQP_pb.addConstr(
                R[arc.j, agent.idx] >= X[arc.idx, agent.idx]
            )

    # 29) Turning off r_i constraints
    for node in nodes:
        for agent in agents:
            MSPP_PD_NQP_pb.addConstr(
                R[node, agent.idx] <= (
                    gb.quicksum(X[arc.idx, agent.idx] for arc in w_arcs if arc.i == node) +
                    gb.quicksum(X[arc.idx, agent.idx]
                                for arc in w_arcs if arc.j == node)
                )
            )

    # 30-32) Well-defined W variable
    for node in nodes:
        for agent in agents:
            for agent_ in agents:
                if agent_.idx < agent.idx:
                    MSPP_PD_NQP_pb.addConstr(
                        W[node, agent.idx, agent_.idx] <= R[node, agent.idx]
                    )
                    MSPP_PD_NQP_pb.addConstr(
                        W[node, agent.idx, agent_.idx] <= R[node, agent_.idx]
                    )
                    MSPP_PD_NQP_pb.addConstr(
                        W[node, agent.idx, agent_.idx] >=
                        R[node, agent.idx] + R[node, agent_.idx] - 1
                    )

    return MSPP_PD_NQP_pb, X, R, W


def set_problem(problem_type, nodes, w_arcs, agents):
    """Formulate the specified optimization problem for a network instance given the agents to route

    Args:
        problem_type (str): The optimization problem to formulate. Only MSPP and MSPP-PD variants are accepted
        nodes (list): list of the nodes in the network instance
        w_arcs (list): list of weighted arcs in the network instance
        agents (list): list of agents that has to be routed

    Returns:
        tuple: a tuple (Problem, *_) where
          - Problem is a gb.Model that represent the selected and created optimization problem
          - *_ is a tuple of gb.MVar containing the decision variables related to the selected problem 
    """

    params = nodes, w_arcs, agents
    if problem_type == "MSPP":
        return set_MSPP(*params)
    elif problem_type == "ABP":
        return set_ABP(*params)
    elif problem_type == "NBP":
        return set_NBP(*params)
    elif problem_type == "ALP":
        return set_ALP(*params)
    elif problem_type == "NLP":
        return set_NLP(*params)
    elif problem_type == "AQP":
        return set_AQP(*params)
    elif problem_type == "NQP":
        return set_NQP(*params)


def evaluate_pb_objectives(problem):
    """Get optimal objectives from an optimization problem

    Args:
        problem (gb.Model): The optimization problem

    Returns:
        list: a list with the optimal values of the different problem's objectives
    """

    # ? Shold I check optimality
    # * See: https://www.gurobi.com/documentation/9.5/refman/working_with_multiple_obje.html
    assert problem.Status == GRB.Status.OPTIMAL

    problem.params.SolutionNumber = 0  # Set best solution found
    opt_solution = []

    # Add to opt_solution the value of each objective
    for obj in range(problem.NumObj):
        problem.params.ObjNumber = obj
        opt_solution.append(problem.ObjNVal)

    return opt_solution
