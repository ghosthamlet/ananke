"""
Class for missing ID
"""

import copy
from ..graphs.admg import ADMG
from ..graphs.dag import DAG
import networkx as nx


class MissingFullID:

    def __init__(self, graph):
        """
        Constructor

        :param graph: Missing data graph to run the ID algorithm on
        """

        self.graph = graph

    def id(self):
        """
        Function to ID the full law

        :return: boolean is ID or not
        """

        for Ri in self.graph.vertices:

            if not Ri.startswith('R_'):
                continue

            #S = set([v for v in self.graph.parents([Ri]) if v.startswith('R_')])
            #S.add(Ri)
            C = set(['X_' + v.split('_')[1] for v in self.graph.parents([Ri]) if v.startswith('R_')])
            C.add('X_' + Ri.split('_')[1])
            F = self.graph.descendants([Ri])
            F.remove(Ri)
            #
            # update = True
            # while update:
            #
            #     update = False
            #     for Rj in F:
            #         if len(self.graph.parents(Rj).intersection(C)) != 0:
            #             update = True
            #             phi_F = copy.deepcopy()
            #             if Rj in


            F = F.difference([f for f in F if len(self.graph.parents([f]).intersection(C)) != 0])

            phi_F = copy.deepcopy(self.graph)
            phi_F.fix(F)
            Ri_indicators = set(['R_' + v.split('_')[1] for v in self.graph.parents([Ri]) if v.startswith('X_')])
            nondescendants_Ri = set(phi_F.vertices).difference(phi_F.descendants([Ri]))
            #print('Ri:', Ri, 'C:', C, 'F:', F, self.graph.parents([Ri]))

            if not Ri_indicators.issubset(nondescendants_Ri) or len(C.intersection(self.graph.parents([Ri]))) != 0:
                return False

        return True

class MissingTargetID:

    def __init__(self, graph):
        """
        Constructor

        :param graph: Missing data graph to run the ID algorithm on
        """

        self.graph = graph
        self.r_subgraph = self.graph.subgraph(self.graph.indicators)
        r_mgraph = nx.DiGraph()
        r_mgraph.add_nodes_from(list(self.r_subgraph.vertices))
        r_mgraph.add_edges_from(list(self.r_subgraph.di_edges))
        self.top_order = list(reversed(list(nx.algorithms.dag.topological_sort(r_mgraph))))

    def identify_propensity_score(self, Ri):
        """
        Identify the propensity score for a particular R_i

        :param R_i: Name of the indicator
        :return: True or False
        """

        descendants_Ri = self.r_subgraph.descendants([Ri])
        ri_subgraph = self.r_subgraph.subgraph(descendants_Ri)
        Gp = nx.DiGraph()
        Gp.add_nodes_from(list(ri_subgraph.vertices))
        Gp.add_edges_from(list(ri_subgraph.di_edges))
        Gp = nx.algorithms.dag.transitive_closure(Gp)
        top_order = list(reversed(list(nx.algorithms.dag.topological_sort(Gp))))[:-1]
        Gp = DAG(vertices=Gp.nodes, di_edges=Gp.edges)
        S = set({Ri})

        updated = True

        while updated:
            zeta_map = {}
            Fi = Gp.children([Ri])
            phi_Fi_G = copy.deepcopy(self.graph)
            phi_Fi_G.fix(Fi)
            S_descendants = S.intersection(phi_Fi_G.descendants([Ri]))
            updated = False

            for Rj in top_order:
                pa_Rj = self.graph.parents([Rj])
                S_descendants_j = S_descendants.intersection(pa_Rj)
                pruned = True

                while pruned:
                    Fj = Gp.children([Rj])
                    phi_Fj_G = copy.deepcopy(self.graph)
                    phi_Fj_G.fix(Fj)
                    descendants_Rj = phi_Fj_G.descendants([Rj])
                    pruned = False

                    for Rk in Fj:
                        zeta_k = zeta_map[Rk]
                        if len(zeta_k.intersection(descendants_Rj.union(S_descendants_j))) != 0:
                            pruned = True
                            Gp.delete_diedge(Rj, Rk)
                            break

                # 􏰗Rk|Xk ∈ paG(Rj)􏰘 ∪ 􏰗􏰊 ∪Rk∈Fj ζk􏰋 ∩ paG(Rj)􏰘
                zeta_map[Rj] = set(['R_' + v.split('_')[1] for v in self.graph.parents([Rj]) if v.startswith('X_')])
                for Rk in Gp.children([Rj]):
                    zeta_map[Rj] = zeta_map[Rj].union(zeta_map[Rk].intersection(pa_Rj))

                phi_Fj_G = copy.deepcopy(self.graph)
                phi_Fj_G.fix(Gp.children([Rj]))
                Rj_indicators = set(['R_' + v.split('_')[1] for v in self.graph.parents([Rj]) if v.startswith('X_')])
                nondescendants_Rj = set(phi_Fj_G.vertices).difference(phi_Fj_G.descendants([Rj]))

                if ((not Rj_indicators.issubset(nondescendants_Rj) or
                     len(zeta_map[Rj].intersection(S_descendants)) != 0) and
                        (Ri, Rj) in Gp.di_edges):
                    Gp.delete_diedge(Ri, Rj)
                    S.add(Rj)
                    updated = True
                    break

        phi_Fi_G = copy.deepcopy(self.graph)
        phi_Fi_G.fix(Gp.children([Ri]))
        print("Final fixing set for", Ri, Gp.children([Ri]))
        print(zeta_map)
        print(Gp.di_edges)
        Ri_indicators = set(['R_' + v.split('_')[1] for v in self.graph.parents([Ri]) if v.startswith('X_')])
        nondescendants_Ri = set(phi_Fi_G.vertices).difference(phi_Fi_G.descendants([Ri]))
        if Ri_indicators.issubset(nondescendants_Ri):
            return True

        return False


    def id(self):
        """
        Function to ID the target law

        :return: Is ID or not
        """

        for R_i in self.top_order:

            if not self.identify_propensity_score(R_i):
                print("Failed to identify", R_i)
                return False

        return True
