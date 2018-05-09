import sys

def train(weights,cacheweights,bias,beta,tagmap,iterations,lines,label):
    keywords=weights.keys()
    count=1
    for iter in range(0,iterations):
        for line in lines:
            tokens=line.split(' ')
            y=tagmap[tokens[label]]
            #print(tokens[label],y,line)
            length=len(tokens)
            existing=set();
            act=0;
            features={};
            for i in range(2,length):
                token=tokens[i]
                if token in keywords and token not in existing:
                    count1=0
                    existing.add(token)
                    for j in range(i,length):
                        if token==tokens[j]:
                            count1+=1
                    features[token]=count1
                    act+=(weights[token]*count1)
            #print(weights)
            act+=bias
            if act * y <= 0:
                #print(tokens[label],y,act,line)
                bias += y
                beta += y*count
                for key in features.keys():
                    weights[key] += features[key] * y
                    cacheweights[key] += features[key] * y * count
            count+=1
            #print(act,bias,act*y,count)
    #print(cacheweights)
    # for (key, val) in cacheweights.items():
    #     cacheweights[key]=weights[key]-(val/count)
    # beta=bias-(beta/count)
    return count

def writemodel(w1,b1,w2,b2,cw1,bt1,cw2,bt2,filename,count):
    modelfile=open(filename,'w')
    modelfile.write('TrueFake')
    modelfile.write('\n')
    if filename=='vanillamodel.txt':
        modelfile.write('bias' + ':' + str(b1))
    else:
        modelfile.write('bias' + ':' + str(round(b1 - (bt1 / count), 2)))
    modelfile.write('\n')
    for (key, val) in w1.items():
        if filename == 'vanillamodel.txt':
            modelfile.write(key + ':' + str(val))
        else:
            modelfile.write(key + ':' + str(round(val-(cw1[key]/count),2)))
        modelfile.write('\n')
    modelfile.write('\n')
    modelfile.write('PosNeg')
    modelfile.write('\n')
    if filename=='vanillamodel.txt':
        modelfile.write('bias' + ':' + str(b2))
    else:
        modelfile.write('bias' + ':' + str(round(b2 - (bt2 / count), 2)))
    modelfile.write('\n')
    for (key, val) in w2.items():
        if filename == 'vanillamodel.txt':
            modelfile.write(key + ':' + str(val))
        else:
            modelfile.write(key + ':' + str(round(val-(cw2[key]/count),2)))
        modelfile.write('\n')


def preprocess(infile):
    processedlines=[]
    vocab=set()
    training=open(infile,'r',encoding='utf-8')
    lines=training.readlines()
    punct = '-`~;:!\\/"?><,.|{}()[]#'
    stopwords=['','the','and','a','to','in','was']
    whitespace = ' ' * len(punct)
    table = str.maketrans(punct, whitespace)
    for line in lines:
        line=line.translate(table)
        processedline = ''
        words=line.split(' ')
        tag1=words[1]
        tag2 = words[2]
        processedline +=tag1+' '+tag2+' '
        for i in range(3,len(words)):
            token=words[i].strip().lower()
            if token not in stopwords:
                vocab.add(token)
                processedline += token + ' '
        processedlines.append(processedline.strip())

    #print(vocab)
    weights1={}
    cacheweights1={}
    bias1=0
    beta1=0
    weights2 = {}
    cacheweights2 = {}
    bias2 = 0
    beta2 = 0
    tagmap = {}
    tagmap['Pos'] = 1
    tagmap['Neg'] = -1
    tagmap['True'] = 1
    tagmap['Fake'] = -1

    for key in vocab:
        weights1[key]=0
        cacheweights1[key]=0
        weights2[key] = 0
        cacheweights2[key] = 0

    #print(bias,beta)
    #print(weights)
    c1=train(weights1,cacheweights1,bias1,beta1,tagmap,15,processedlines,0)
    #print(c1)
    c2=train(weights2,cacheweights2,bias2,beta2,tagmap,20,processedlines,1)
    #print(c2)
    writemodel(weights1,bias1,weights2,bias2,cacheweights1,beta1,cacheweights2,beta2,'vanillamodel.txt',c1)
    # print(weights1)
    # print(weights2)
    writemodel(weights1,bias1,weights2,bias2,cacheweights1,beta1,cacheweights2,beta2,'averagedmodel.txt',c2)


def main(arg):
    preprocess(arg)

if __name__=="__main__":
    #print(time.time())
    main(sys.argv[1])
    #main("train-labeled.txt")
    #print(time.time())
