
import scimple as scm

# example :


molecTable = scm.Table("../scimple/scimple_data/phenyl-Fe-porphyirine-CO2-Me_4_rel.xyz", firstLine=3, lastLine=103)
grapheneTable = scm.Table("../scimple/scimple_data/phenyl-Fe-porphyirine-CO2-Me_4_rel.xyz", firstLine=104, lastLine=495)
chargesGraphene = scm.Table("../scimple/scimple_data/CHARGES_phenyl-Fe-porphyirine-CO2-Me_4_rel", firstLine=104, lastLine=495)
print(molecTable.getTable())

# 3D delta et molec

myPlot3D = scm.Plot(dim=3, xlabel="X", ylabel="Y", zlabel="Z", borders=[-40, 40, -40, 40, 15, 30],
                              title="Test Graphe #3D delta et molec")
myPlot3D.add(molecTable, xColNum=2, yColNum=3, zColNum=4, markersize=2, coloredBy=1)
myPlot3D.add(grapheneTable, xColNum=2, yColNum=3, zColNum=4, markersize=2, label="graphene",
                   coloredBy=lambda lineNum, line: (sum(chargesGraphene.getTable()[lineNum][1:]) - 4))
"""EN TESTS :
#3D molec avec couleurs standards
dicoCouleursStandards={'C':"#000000",'H':"#ffffff",'O':'r','N':'b','Fe':"#00ffff"}
myPlot3D=Plot(dim=3,xlabel="X",ylabel="Y",zlabel="Z",borders=[-40,40,-40,40,15,30],title="Test Graphe #3D molec avec couleurs standards")
myPlot3D.add(molecTable,xColNum=2,yColNum=3,zColNum=4,coloredBy=lambda lineNum,line:dicoCouleursStandards[line[1]])
"""
# 3D comparatif z et delta:

myPlot3Dbis = scm.Plot(dim=3, xlabel="X", ylabel="Y", zlabel="Z", borders=[-40, 40, -40, 40, 15, 30],
                                 title="Test Graphe #3D comparatif z et delta:")
myPlot3Dbis.add([grapheneTable.getTable()[i][:4] + [grapheneTable.getTable()[i][4] + 10] for i in
                       range(len(grapheneTable.getTable()) - 1)], xColNum=2, yColNum=3, zColNum=4, label="colored by z",
                      coloredBy=lambda lineNum, line: line[4])
myPlot3Dbis.add(grapheneTable, xColNum=2, yColNum=3, zColNum=4, label="colored by delta",
                      coloredBy=lambda lineNum, line: (sum(chargesGraphene.getTable()[lineNum][1:]) - 4))

# 2D:

myPlot2D = scm.Plot(dim=2, xlabel="X", zlabel="Z", borders=[-20, 20, 18, 19], title="Test Graphe 2D")
myPlot2D.add(grapheneTable, xColNum=3, yColNum=4, label="graphene Y/Z", coloredBy="#f4a28c", markersize=20)
myPlot2D.add(grapheneTable, xColNum=2, yColNum=4, label="graphene X/Z", plotType='-')

# 3D plot 2 surfaces:

myTable = scm.Table("../scimple/scimple_data/ek_InTP_CO2_Me_4_graphene_W_r2_k.dat", firstLine=1)

myPlot3Dter = scm.Plot(dim=3, xlabel="X", ylabel="Y", zlabel="Z", title="deux surfaces, point de weyl ?")
myPlot3Dter.add(myTable, xColNum=0, yColNum=1, zColNum=4, label="column 4", coloredBy="#000000")
myPlot3Dter.add(myTable, xColNum=0, yColNum=1, zColNum=5, label="column 5")
scm.showAndBlock()
