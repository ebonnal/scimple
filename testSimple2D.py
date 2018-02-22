import scimple.scimple as scimple
grapheneTable=scimple.ImportTable("test.xyz",firstLine=104,lastLine=495)
myPlot2D=scimple.CreatePlot(2,xlabel="X",zlabel="Z",borders=[-20,20,18,19],title="Test Graphe 2D")
myPlot2D.addToPlot(grapheneTable,3,4,label="graphene Y/Z",color="#ff0000")
myPlot2D.addToPlot(grapheneTable,2,4,label="graphene X/Z",color="#ffff00")