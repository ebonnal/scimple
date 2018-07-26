import pandas as pd
import re
from math import ceil
from .plot import Plot
# #####
# STATS
# #####


class Serie:
    valeurs = None  # list
    length = 0  # int
    # attributes:
    _attr_moyenne = 'moyenne'
    _attr_mediane = 'mediane'
    _attr_tableau_statistique = 'tableau_statistique'
    _attr_description = 'description'

    # types:
    type = None
    qualitative_nominale = 0
    qualitative_ordinale = 1
    quantitative_discrete = 2
    quantitative_continue = 3

    def __init__(self, valeurs, type_):
        self.valeurs = list(valeurs)
        self.valeurs.sort()
        self.length = len(self.valeurs)
        self.type = type_

    def __getattr__(self, item):
        if item == Serie._attr_moyenne:
            if self.type in {Serie.quantitative_discrete, Serie.quantitative_continue}:
                return sum(self.valeurs) / self.length
            else:
                raise AttributeError("mean is only computable on quantitative series")

        if item == Serie._attr_mediane:
            if self.type in {Serie.quantitative_discrete, Serie.quantitative_continue}:
                return self.valeurs[(self.length + 1) // 2] \
                       if self.length % 2 else (self.valeurs[self.length // 2] + self.valeurs[self.length // 2 + 1]) / 2
            else:
                raise AttributeError("mean is only computable on quantitatives series")

        match = re.fullmatch(r'quantile_(\d+)_(\d+)', item)
        if match is not None:
            if self.type in {Serie.quantitative_discrete, Serie.quantitative_continue}:
                p = int(match.group(1))/int(match.group(2))
                return self.valeurs[int(ceil(self.length * p))] \
                       if int(self.length*p) != self.length*p else \
                    (self.valeurs[int(self.length * p)] + self.valeurs[int(self.length * p) + 1]) / 2
            else:
                raise AttributeError("quantiles are only computable on quantitatives series")

        elif item == Serie._attr_tableau_statistique:
            if self.type != Serie.quantitative_continue:
                valeurs_uniques = set(self.valeurs)
                res = pd.DataFrame([[self.n(valeur_unique), self.f(valeur_unique)]
                                    for valeur_unique in valeurs_uniques])
                res.index = pd.Series(list(map(lambda e: f'valeur {e}', list(valeurs_uniques)))
                                      , name='grandeurs')
                res.columns = ['n', 'f']
                res.sort_values(['f'], inplace=True, ascending=False)
                return res
            else:
                raise AttributeError("tableau statistique is only computable on quantitative_discrete or qualitatives series")

        elif item == Serie._attr_description:
            if self.type != Serie.quantitative_continue:
                print(self.tableau_statistique)
            if self.type in {Serie.quantitative_discrete, Serie.quantitative_continue}:
                print()
                print(f"Moyenne={self.mediane}\n"
                      f"1er Quartile={self.quantile_1_4}\n"
                      f"Mediane={self.mediane}\n"
                      f"3eme Quartile={self.quantile_3_4}\n")
            Plot(2, 'boxplot').boxplot(self.valeurs, labels=[''])
        else:
            raise AttributeError(f"{item} is not a valid attribute")

    def n(self, value):
        return sum([value == current for current in self.valeurs])

    def f(self, value):
        return self.n(value)/self.length


if __name__ == "__main__":
    s = Serie([1,2,3,4,5,6,7,8,8,8,8,8,9,9,9,9,1,0,1,0,1,10], Serie.quantitative_discrete)
    print(s.length,s.moyenne,s.mediane,
          s.quantile_10_400, s.quantile_2_4, s.quantile_3_4)
    print(s.tableau_statistique)
