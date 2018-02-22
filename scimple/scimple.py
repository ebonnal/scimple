from __future__ import absolute_import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.interpolate as interp
import warnings
from random import randint
warnings.filterwarnings("ignore")


def randomColor(racinecubiquesup,pas):
    R=hex(randint(0,racinecubiquesup-1)*pas)[2:]
    if len(R)==1:
        R="0"+R
    G=hex(randint(0,racinecubiquesup-1)*pas)[2:]
    if len(G)==1:
        G="0"+G
    B=hex(randint(0,racinecubiquesup-1)*pas)[2:]
    if len(B)==1:
        B="0"+B
    while R==B and R==G:
        print(3)

        R=hex(randint(0,racinecubiquesup-1)*pas)[2:]
        if len(R)==1:
            R="0"+R
        G=hex(randint(0,racinecubiquesup-1)*pas)[2:]
        if len(G)==1:
            G="0"+G
        B=hex(randint(0,racinecubiquesup-1)*pas)[2:]
        if len(B)==1:
            B="0"+B
    return "#"+R+G+B
class CreatePlot:
    def __init__(self,dim=2,title="",borders=None,xlabel="",ylabel="",zlabel=""):
        self.__atLeastOneLabelDefined=False
        plt.rcParams['lines.color'] = 'b'
        if dim==2:
            plt.title(title)
            if type(borders)==list:
                if len(borders)==4:
                    plt.xlim(borders[0],borders[1])
                    plt.ylim(borders[2],borders[3])      
                else:
                    print("length of borders list for 2D plot must be 4")
                    raise Exception()
        elif dim==3:
            self.__fig = plt.figure()
            self.__ax = self.__fig.gca(projection='3d')
            self.__ax.set_title(title)
            self.__ax.set_xlabel(xlabel)
            self.__ax.set_ylabel(ylabel)
            self.__ax.set_zlabel(zlabel)
            if type(borders)==list:
                if len(borders)==6:
                    self.__ax.set_xlim(borders[0],borders[1])
                    self.__ax.set_ylim(borders[2],borders[3])
                    self.__ax.set_zlim(borders[4],borders[5])        
                else:
                    print("length of borders list for 3D plot must be 6")
                    raise Exception()
        else:
            print("SCIMPLE ERROR : in Plot(dim), dim must be 2 or 3")
            raise Exception()
        self.__dim=dim#string
        self.__plotables=[]
    
    def addToPlot(self,tableOrImportedTable,xColNum,yColNum,zColNum=None,label=""\
    ,color=None,coloredBy=None,plotType='o',markersize=9):
        if type(tableOrImportedTable)==list:
            table=tableOrImportedTable
        elif type(tableOrImportedTable)==ImportTable:
            table=tableOrImportedTable.getTable()
        else :
            print("SCIMPLE ERROR : table format not supported")
        
        if self.__dim==2:
            if zColNum!=None:
                print("SCIMPLE ERROR : z column declsaration for 2D plot forbidden")
                raise Exception()
            if label!="":
                self.__atLeastOneLabelDefined=True
            X,Y=[],[]
            for lineIndex in range(0,len(table)):
                if len(table[lineIndex])>max(xColNum,yColNum) and\
                table[lineIndex][xColNum]!=None and \
                table[lineIndex][yColNum]!=None:
                    X.append(table[lineIndex][xColNum])
                    Y.append(table[lineIndex][yColNum])
            if coloredBy !=None:
                plt.plot(X,Y,plotType,label=label,color=coloredBy,markersize=markersize)
            else:
                plt.plot(X,Y,plotType,label=label,markersize=markersize)

            if self.__atLeastOneLabelDefined:
                plt.legend(loc='upper right', shadow=True).draggable() 
            plt.show()
          
        else:
            print(coloredBy)
            if zColNum==None:
                print("SCIMPLE ERROR : z column declaration required for 3D plot")
                raise Exception()
            if type(coloredBy)==int:#INT COLNUM
                self.__atLeastOneLabelDefined=True
                #build groupsDico
                groupsDic={}
                for line in table:
                    if line[coloredBy] in groupsDic:
                        groupsDic[line[coloredBy]]+=[line]
                    else:
                        groupsDic[line[coloredBy]]=[line]
                racinecubiquesup=0
                while racinecubiquesup**3-racinecubiquesup<=len(groupsDic):
                    print(1)
                    racinecubiquesup+=1
                pas=255//(racinecubiquesup-1)
                listOfUsedColors=[]
                for group in groupsDic:
                    table=groupsDic[group]
                    X,Y,Z=[],[],[]
                    for lineIndex in range(0,len(table)):
                        if len(table[lineIndex])>max(xColNum,yColNum,zColNum) and\
                        table[lineIndex][xColNum]!=None and \
                        table[lineIndex][yColNum]!=None and \
                        table[lineIndex][zColNum]!=None:
                            X.append(table[lineIndex][xColNum])
                            Y.append(table[lineIndex][yColNum])
                            Z.append(table[lineIndex][zColNum])
                    groupColor=randomColor(racinecubiquesup,pas)
                    while groupColor in listOfUsedColors:
                        print(2)

                        groupColor=randomColor(racinecubiquesup,pas)
                    listOfUsedColors.append(groupColor)
                    self.__ax.plot(X,Y,Z,plotType,label=str(group),color=groupColor,markersize=markersize)
            elif str(type(coloredBy))=="<class 'function'>" :#and type(coloredBy(1,table[0]))==int :#lineNum,lineList -> int indicateur
                
                maxi=None
                mini=None
                for i in range(len(table)):
                    try:
                        value=coloredBy(i,table[i])
                        if value ==-4:
                            print(table[i])
                    except:
                        value=maxi
                    if maxi==None:
                        maxi=value
                        mini=value
                    try:
                        maxi=max(maxi,value)
                        mini=min(mini,value)
                    except :
                        pass
                if label!="":
                    self.__atLeastOneLabelDefined=True
                colorDico={} #hexa -> plotable lines
                for lineIndex in range(0,len(table)):
                    if len(table[lineIndex])>max(xColNum,yColNum,zColNum) and\
                    table[lineIndex][xColNum]!=None and \
                    table[lineIndex][yColNum]!=None and \
                    table[lineIndex][zColNum]!=None:
                        colorRes=coloredBy(lineIndex,table[lineIndex])
                        deux=colorRes-mini
                        maxolo=max(0,deux)
                        minolo=min(255,maxolo*255/(maxi-mini))
                        colorHexaUnit=hex(int(minolo))[2:]
                        if len(colorHexaUnit)==1:
                            colorHexaUnit="0"+colorHexaUnit
                        color="#"+colorHexaUnit*3
                        if color in colorDico:
                            colorDico[color][0]+=[table[lineIndex][xColNum]]
                            colorDico[color][1]+=[table[lineIndex][yColNum]]
                            colorDico[color][2]+=[table[lineIndex][zColNum]]
                        else:
                            colorDico[color]=[[table[lineIndex][xColNum]],\
                            [table[lineIndex][yColNum]],[table[lineIndex][zColNum]]]
                         
                legendOn=True
                for colorGroup in colorDico:
                        self.__ax.plot(colorDico[colorGroup][0],colorDico[colorGroup][1],\
                        colorDico[colorGroup][2],plotType,label=(label if legendOn else ""),color=colorGroup,markersize=markersize,solid_capstyle="round")
                        legendOn=False
                
                
            elif type(coloredBy)==str or coloredBy==None:#simple color field provided or nothing
                if label!="":
                    self.__atLeastOneLabelDefined=True
                X,Y,Z=[],[],[]
                for lineIndex in range(0,len(table)):
                    if len(table[lineIndex])>max(xColNum,yColNum,zColNum) and\
                    table[lineIndex][xColNum]!=None and \
                    table[lineIndex][yColNum]!=None and \
                    table[lineIndex][zColNum]!=None:
                        X.append(table[lineIndex][xColNum])
                        Y.append(table[lineIndex][yColNum])
                        Z.append(table[lineIndex][zColNum])
                if coloredBy !=None:
                    self.__ax.plot(X,Y,Z,plotType,label=label,color=coloredBy,markersize=markersize)
                else:
                    self.__ax.plot(X,Y,Z,plotType,label=label,markersize=markersize)
            else:
                print(coloredBy)
                print("color argument must be function int,List->string ,or string, or int")
                raise Exception()
                
            if self.__atLeastOneLabelDefined:
                    self.__ax.legend(loc='upper right', shadow=True).draggable()    

class ImportTable:
    def __init__(self,path,firstLine=1,lastLine=None,columnNames=None\
    ,delimiter=r'(\t|[ ])+',newLine=r'(\t| )*((\r\n)|\n)',floatDot='.',numberFormatCharacter='',ignore="",\
    printTokens=False,printError=False):
        #dev args:
        self.__printTokens=printTokens
        self.__printError=printError
        #init fields
        self.__path=path#string
        self.__firstLine=firstLine#int
        self.__lastLine=lastLine#int
        self.__columnNames=columnNames#list
        self.__contentAsString=None#string
        self.__floatTable=[]#string
        self.__delimiter=delimiter#regExp
        self.__newLine=newLine#regExp
        self.__floatDot=(r'\.' if floatDot=='.' else floatDot)#regExp
        self.__numberFormatCharacter=numberFormatCharacter#string
        self.__ignore=ignore#regExp
        #import file
        try:
            inFile=open(path,'r')
            self.__contentAsString=inFile.read()
            inFile.close()
            self.__parse()
        except IOError as e:
            print("SCIMPLE ERROR : le chemin "+path+" est introuvable :("+\
                "I/O error({0}): {1}".format(e.errno, e.strerror))
    
    def __parse(self):
        # List of token names. 
        tokens = (
            'float',
            'delimiter',
            'newLine',
            'string'
        )
        #variable :
        lineNumber=0
        #Regles :
        
        def t_newLine(t):
            r''
            t.lexer.lineno+=1
            return t
        t_newLine.__doc__=self.__newLine
        def t_delimiter(t):
            r''
            return t
        t_delimiter.__doc__=self.__delimiter
        def t_float(t):
            r''
            t.value=t.value.replace(self.__numberFormatCharacter,'')
            if self.__floatDot!='\.':
                t.value=t.value.replace(self.__floatDot,'.')
            try:
                t.value=float(t.value)
                return t
            except:
                t.lexer.skip(1)
            
        t_float.__doc__=r'(-|[0-9])+('+self.__floatDot+'[0-9]*)?'
        t_string=r'([a-z]|[A-Z]|_)([a-z]|[A-Z]|_|-|[0-9])*'
        t_ignore  =self.__ignore
        def t_eof(t):
            self.__lastLine=-1
            t.type='newLine'
            return t
        # en cas d'ERROR :
        def t_error(t):
            if self.__printError:
                print("Error on char : '%s'" % t.value[0])#dev
            t.lexer.skip(1)
        # Build du lexer
        lexer = lex.lex()
        # On donne l'input au lexer
        lexer.input(self.__contentAsString)
        # On build la string résultat :
        tok = lexer.token()
        if self.__printTokens:
            print(tok)
        currentLine=list()
        currentFloat=None
        while tok:
            if tok.lineno>=self.__firstLine and (self.__lastLine==None or tok.lineno<=self.__lastLine):
                if tok.type=="newLine":
                    currentLine.append(currentFloat)
                    currentFloat=None

                    self.__floatTable.append(currentLine)
                    currentLine=[]
                elif tok.type=="float" or tok.type=="string":
                    currentFloat=tok.value
                elif tok.type=="delimiter":
                    currentLine.append(currentFloat)
                    currentFloat=None
            elif self.__lastLine!=None and tok.lineno>self.__lastLine:
                break
            tok = lexer.token()
            if self.__printTokens:
                print(tok)
        if not(tok):
            currentLine.append(currentFloat)
            self.__floatTable.append(currentLine)
    #public :
    def getTable(self):
        """returns the table (list of list) of floats with None for empty fields"""
        return self.__floatTable
    def getString(self):
        return self.__contentAsString

if __name__=='__main__':
    import ply.lex as lex
    mydata=ImportTable("test.txt")
    #mydata=ImportTable(firstLine=1,lastLine=10,delimiter=r"\n",newLine="jhiotioh",ignore=" \t")
else :
    from .ply import lex

