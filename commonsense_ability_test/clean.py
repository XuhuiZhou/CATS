import random
with open("wsc.txt", "r") as f:
    file = f.readlines()
num = len(file)

t_dic = {'A':0, 'B':1, 'C':2}

with open("wsc_test.txt", "w") as f:
    for i in file:
        label, sentence_1, sentence_2 = i.split("\001")
        label = t_dic[label]
        f.write('{}'.format(label)+'\001'+sentence_1+'\001'+sentence_2)

        #label = 0
        '''
        label = random.randint(0, 1)
        if label==0:
            f.write('{}'.format(label)+'\001'+sentence_1.strip()+'\001'+sentence_2)
        else:
            f.write('{}'.format(label)+'\001'+sentence_2.strip()+'\001'+sentence_1+'\n')
        '''


