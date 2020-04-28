"""
Class for one line ID algorithms.
"""

import copy
import os


class NotIdentifiedError(Exception):
    """
    Custom error for when desired functional is not identified.
    """
    pass


class OneLineID:

    def __init__(self, graph, treatments, outcomes):
        """
        Applies the ID algorithm (Shpitser and Pearl, 2006) reformulated in a 'one-line' fashion (Richardson et al., 2017).

        :param graph: Graph on which the query will run.
        :param treatments: iterable of names of variables being intervened on.
        :param outcomes: iterable of names of variables whose outcomes we are interested in.
        """

        self.graph = graph
        self.treatments = [A for A in treatments]
        self.outcomes = [Y for Y in outcomes]
        self.swig = copy.deepcopy(graph)
        self.swig.fix(self.treatments)
        self.ystar = {v for v in self.swig.ancestors(self.outcomes) if not self.swig.vertices[v].fixed}
        self.Gystar = self.graph.subgraph(self.ystar)
        # dictionary mapping the fixing order for each p(D | do(V\D) )
        self.fixing_orders = {}

    def draw_swig(self, direction=None):
        """
        Draw the proper SWIG corresponding to the causal query.

        :return: dot language representation of the SWIG.
        """

        swig = copy.deepcopy(self.graph)

        # add fixed vertices for each intervention
        for A in self.treatments:

            fixed_vertex_name = A.lower()
            swig.add_vertex(fixed_vertex_name)
            swig.vertices[fixed_vertex_name].fixed = True

            # delete all outgoing edges from random vertex
            # and give it to the fixed vertex
            for edge in swig.di_edges:
                if edge[0] == A:
                    swig.delete_diedge(edge[0], edge[1])
                    swig.add_diedge(fixed_vertex_name, edge[1])

            # finally, add a fake edge between interventions and fixed vertices
            # just for nicer visualization
            swig.add_diedge(A, A.lower())

        return swig.draw(direction)

    def id(self):
        """
        Run one line ID for the query.

        :return: boolean that is True if p(Y(a)) is ID, else False.
        """

        self.fixing_orders = {}
        vertices = set(self.graph.vertices)

        # check if each p(D | do(V\D) ) corresponding to districts in Gystar is ID
        for district in self.Gystar.districts:

            fixable, order = self.graph.fixable(vertices - district)

            # if any piece is not ID, return not ID
            if not fixable:
                return False

            self.fixing_orders[tuple(district)] = order

        return True

    # TODO try to reduce functional
    def functional(self):
        """
        Creates and returns a string for identifying functional.

        :return: string representing the identifying functional.
        """

        if not self.id():
            raise NotIdentifiedError

        # create and return the functional
        functional = '' if set(self.ystar) == set(self.outcomes) else '\u03A3'

        for y in self.ystar:
            if y not in self.outcomes:
                functional += y
        if len(self.ystar) > 1: functional += ' '

        print(self.fixing_orders)
        for district in sorted(list(self.fixing_orders)):
            functional += '\u03A6' + ''.join(reversed(self.fixing_orders[district])) + '(p(V);G) '
        return functional

    # TODO export intermediate CADMGs for visualization
    def export_intermediates(self, folder="intermediates"):
        """
        Export intermediate CADMGs obtained during fixing.

        :param folder: string specifying path to folder where the files will be written.
        :return: None.
        """

        # make the folder if it doesn't exist
        if not os.path.exists(folder):
            os.mkdir(folder)

        # clear the directory
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        # do the fixings and render intermediate CADMGs
        for district in self.fixing_orders:

            G = copy.deepcopy(self.graph)

            fixed_vars = ""
            dis_name = "".join(district)

            for v in self.fixing_orders[district]:
                fixed_vars += v
                G.fix(v)
                G.draw().render(os.path.join(folder,
                                             "phi" + fixed_vars + "_dis" + dis_name + ".gv"))


def get_required_intrinsic_sets(admg):
    required_intrinsic_sets, _ = admg.get_intrinsic_sets()
    return required_intrinsic_sets


def get_allowed_intrinsic_sets(experiments):
    allowed_intrinsic_sets = set()
    allowed_intrinsic_dict = dict()
    fixing_orders = dict()
    for index, experiment in enumerate(experiments):
        intrinsic_sets, order_dict = experiment.get_intrinsic_sets()
        allowed_intrinsic_sets.update(intrinsic_sets)
        fixing_orders[index] = order_dict
        for s in intrinsic_sets:
            allowed_intrinsic_dict[frozenset(s)] = index
    return allowed_intrinsic_sets, allowed_intrinsic_dict, fixing_orders


class OneLineGID:

    def __init__(self, graph, interventions, outcomes):
        """
        Applies the naive one-line GID algorithm.

        :param graph: Graph on which the query will be run.
        :param interventions: Iterable of treatment variables.
        :param outcomes: Iterable of outcome variables.
        """
        self.graph = graph
        self.interventions = interventions
        self.outcomes = outcomes
        self.swig = copy.deepcopy(graph)
        self.swig.fix(self.interventions)
        self.ystar = {v for v in self.swig.ancestors(self.outcomes) if not self.swig.vertices[v].fixed}
        self.Gystar = self.graph.subgraph(self.ystar)

    def _allowed_intrinsic_sets(self, experiments):
        allowed_intrinsic_sets = set()
        allowed_intrinsic_dict = dict()
        fixing_orders = dict()
        for experiment in experiments:
            swig = copy.deepcopy(self.graph)
            swig.fix(experiment)
            intrinsic_sets, order_dict = swig.get_intrinsic_sets()
            allowed_intrinsic_sets.update(intrinsic_sets)
            fixing_orders[frozenset(experiment)] = order_dict
            for s in intrinsic_sets:
                allowed_intrinsic_dict[frozenset(s)] = experiment
        return allowed_intrinsic_sets, allowed_intrinsic_dict, fixing_orders

    def functional(self, experiments=[set()]):
        """
        Creates a string representing the identifying functional.

        :param experiments: A list of sets denoting the interventions of the available experimental distributions.
        :return: string representing the identifying functional.
        """
        if not self.id(experiments=experiments):
            raise NotIdentifiedError

        # create and return the functional
        functional = '' if set(self.ystar) == set(self.outcomes) else '\u03A3'

        for y in self.ystar:
            if y not in self.outcomes:
                functional += y
        if len(self.ystar) > 1: functional += ' '

        # guarantee a deterministic printing order
        fixing = []
        items = []
        for item in self.required_intrinsic_sets:
            fixed = self.allowed_intrinsic_dict[item]
            fixing.append(list(fixed))
            items.append(item)

        sorted_items = sorted(items, key=dict(zip(items, fixing)).get)
        sorted_fixing = sorted(fixing)

        for i, item in enumerate(sorted_items):
            fixed = sorted_fixing[i]

            correct_order = self.fixing_orders[frozenset(fixed)][frozenset(item) - frozenset(fixed)]

            functional += '\u03A6' + ','.join(reversed(correct_order)) + ' p(V \\ {0} | do({0}))'.format(
                ",".join(fixed))

        return functional

    def id(self, experiments=[set()]):
        """
        Checks if identification query is identified given the set of experimental distributions/

        :param experiments: A list of sets denoting the interventions of the available experimental distributions.
        :return: boolean indicating if query is ID or not.
        """
        required_intrinsic_sets = get_required_intrinsic_sets(self.Gystar)
        allowed_intrinsic_sets, allowed_intrinsic_dict, fixing_orders = self._allowed_intrinsic_sets(experiments)
        self.required_intrinsic_sets = required_intrinsic_sets
        self.allowed_intrinsic_sets = allowed_intrinsic_sets
        self.allowed_intrinsic_dict = allowed_intrinsic_dict
        self.fixing_orders = fixing_orders

        is_id = False

        if allowed_intrinsic_sets >= required_intrinsic_sets:
            is_id = True

        return is_id


def check_experiments_ancestral(admg, experiments):
    """
    Check that each experiment G(S(b_i)) is ancestral in ADMG G(V(b_i))
    https://simpleflying.com/

    :param admg: An ADMG
    :param experiments: A list of ADMGs representing experiments
    :return:
    """
    for experiment in experiments:
        graph = copy.deepcopy(admg)
        fixed = experiment.fixed
        graph.fix(fixed)
        if not experiment.is_ancestral_subgraph(admg):
            return False

    return True


class OnelineAID:

    def __init__(self, graph, treatments, outcomes):
        """
        Applies the one-line AID algorithm.

        :param graph: Graph on which the query will be run
        :param treatments: Iterable of treatment variables
        :param outcomes: Iterable of outcome variables
        """
        self.graph = graph
        self.interventions = treatments
        self.outcomes = outcomes
        self.swig = copy.deepcopy(graph)
        self.swig.fix(self.interventions)
        self.ystar = {v for v in self.swig.ancestors(self.outcomes) if not self.swig.vertices[v].fixed}
        self.Gystar = self.graph.subgraph(self.ystar)

        self.checked_id = False

    def id(self, experiments):
        if not check_experiments_ancestral(admg=self.graph, experiments=experiments):
            raise NotIdentifiedError
        self.required_intrinsic_sets = get_required_intrinsic_sets(admg=self.Gystar)
        self.allowed_intrinsic_sets, self.allowed_intrinsic_dict, self.fixing_orders = get_allowed_intrinsic_sets(
            experiments=experiments)

        is_id = False
        if self.allowed_intrinsic_sets >= self.required_intrinsic_sets:
            is_id = True

        self.checked_id = True

        return is_id

    def functional(self, experiments):
        """
        Creates a string representing the identifying functional

        :param experiments: A list of sets denoting the interventions of the available experimental distributions
        :return:
        """
        if not check_experiments_ancestral(admg=self.graph, experiments=experiments):
            raise NotIdentifiedError
        if not self.id(experiments=experiments):
            raise NotIdentifiedError

        # create and return the functional
        functional = '' if set(self.ystar) == set(self.outcomes) else '\u03A3'

        for y in self.ystar:
            if y not in self.outcomes:
                functional += y
        if len(self.ystar) > 1: functional += ' '

        # guarantee a deterministic printing order
        fixing = []
        intrinsic_sets = []
        for intrinsic_set in self.required_intrinsic_sets:
            fixed = experiments[self.allowed_intrinsic_dict[intrinsic_set]].fixed
            fixing.append(list(fixed))
            intrinsic_sets.append(intrinsic_set)

        sorted_intrinsic_sets = sorted(intrinsic_sets, key=dict(zip(intrinsic_sets, fixing)).get)
        sorted_fixing = sorted(fixing)

        for i, intrinsic_set in enumerate(sorted_intrinsic_sets):
            fixed = sorted_fixing[i]
            vars = sorted(
                set([v for v in experiments[self.allowed_intrinsic_dict[intrinsic_set]].vertices]) - set(fixed))
            correct_order = self.fixing_orders[self.allowed_intrinsic_dict[intrinsic_set]][
                frozenset(intrinsic_set) - frozenset(fixed)]
            if len(correct_order):
                functional += '\u03A6' + ','.join(reversed(correct_order))
            functional += ' p({0} | do({1}))'.format(",".join(vars), ",".join(fixed))

        return functional
