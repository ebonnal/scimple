import pandas as pd
# #####
# STATS
# #####


class Serie:
    values = None  # list
    length = 0  # int
    # attributes:
    _attr_moyenne = 'moyenne'
    _attr_tableau_statistique = 'tableau_statistique'
    # types:
    type = None
    qualitative_nominale = 0
    qualitative_ordinale = 1
    quantitative_discrete = 2
    quantitative_continue = 3

    def __init__(self, values, type_):
        self.values = list(values)
        self.length = len(self.values)
        self.type = type_

    def __getattr__(self, item):
        if item == Serie._attr_moyenne:
            if self.type in {Serie.quantitative_discrete, Serie.quantitative_continue}:
                return sum(self.values)/self.length
            else:
                raise AttributeError("mean is only computable on quantitative series")

        elif item == Serie._attr_tableau_statistique:
            valeurs_uniques = set(self.values)
            res = pd.DataFrame([[self.n(valeur_unique), self.f(valeur_unique)]
                                for valeur_unique in valeurs_uniques])
            res.index = pd.Series(list(valeurs_uniques), name='valeurs')
            res.columns = ['n', 'f']
            return res

    def n(self, value):
        return sum([value == current for current in self.values])

    def f(self, value):
        return self.n(value)/self.length
