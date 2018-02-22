def removeRegExp(content,regExpToRemove):
    # List of token names. 
    tokens = (
        'toRemove',
        'reste',
    )
    def t_toRemove(t):
        r"[0-9]{10}|[0-9]{9}"
    if regExpToRemove!=None:
        t_toRemove.__doc__=regExpToRemove
    def t_reste(t):
        r'.|\n'
        return t
    def t_eof(t):
        print("End Of File reached")
    # en cas d'ERROR :
    def t_error(t):
        print("Error on char : '%s'" % t.value[0])#dev
        t.lexer.skip(1)
    # Build du lexer
    lexer = lex.lex()
    # On donne l'input au lexer
    lexer.input(content)
    # On build la string résultat :
    tok = lexer.token()
    str_res=""
    while tok:
        print(tok.value)
        str_res+=tok.value
        tok = lexer.token()
    print(str_res+"###")
    return str_res
        
if __name__=='__main__':
    import ply.lex as lex
    mydata=ImportTable("test.txt")
    #mydata=ImportTable(firstLine=1,lastLine=10,delimiter=r"\n",newLine="jhiotioh",ignore=" \t")
else :
    from .ply import lex