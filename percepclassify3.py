import sys

def processdev(modelfile):
    model=open(modelfile,'r')
    model.readline()
    bias=model.readline()
    bias1=float(bias.split(':')[1].strip())
    truefake=model.readline()
    weights1={}
    while truefake!='\n':
        wordcount=truefake.split(':')
        weights1[wordcount[0]]=float(wordcount[1].strip())
        truefake=model.readline()
    model.readline()
    bias = model.readline()
    bias2 = float(bias.split(':')[1].strip())
    posneg = model.readline()
    weights2 = {}
    while posneg != '':
        wordcount = posneg.split(':')
        weights2[wordcount[0]] = float(wordcount[1].strip())
        posneg = model.readline()
    return weights1,weights2, bias1, bias2

def decode(weights1,weights2,b1,b2,test):
    devtextfile = open(test, encoding='utf-8')
    lines = devtextfile.readlines()
    punct = '-`~;:!\\/"?><,.|{}()[]#'
    stopwords=['','the','and','a','to','in','was']
    whitespace = ' ' * len(punct)
    table = str.maketrans(punct, whitespace)
    outfile=open('percepoutput.txt','w',encoding='utf-8')
    keywords1 = weights1.keys()
    keywords2 = weights2.keys()
    for line in lines:
        line = line.translate(table)
        words = line.split(' ')
        existing1 = set();
        existing2 = set();
        act1 = 0;
        act2 = 0;
        features1 = {};
        features2 = {};
        length = len(words)
        for i in range(1, len(words)):
            token=words[i].strip().lower()
            if token in stopwords:
                continue
            if token in keywords1 and token not in existing1:
                count1 = 0
                existing1.add(token)
                for j in range(i, length):
                    if token == words[j].lower():
                        count1 += 1
                features1[token] = count1
                act1 += (weights1[token] * count1)
            if token in keywords2 and token not in existing2:
                count2 = 0
                existing2.add(token)
                for j in range(i, length):
                    if token == words[j].lower():
                        count2 += 1
                features2[token] = count2
                act2 += (weights2[token] * count2)
        act1 += b1
        act1 += b2
        # print(act1,act2)
        tag1 = 'Fake'
        tag2 = 'Neg'
        if (act1 > 0):
            tag1 = 'True'
        if act2 > 0:
            tag2 = 'Pos'
        outfile.write(words[0] + ' ' + tag1 + ' ' + tag2)
        outfile.write('\n')

def main(model,test):
    weight1,weight2,b1,b2=processdev(model)
    decode(weight1,weight2,b1,b2,test)

if __name__=="__main__":
    #print(time.time())
    main(sys.argv[1],sys.argv[2])
    # main("vanillamodel.txt","dev-text.txt")
    #main("averagedmodel.txt","dev-text.txt")
