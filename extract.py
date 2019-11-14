import csv
data_2 = []
data_3 = []
with open('aug_arct.csv', 'r') as fp:
    reader = csv.reader(fp, delimiter=',', quotechar='"')
    for row in reader:
        try:
            print(row[-1])
            if int(row[-1])==3:
                data_3.append([row[0],row[1], row[2], row[5], row[6], row[7]])
            elif int(row[-1])==2:
                data_2.append([row[0],row[1], row[2], row[5], row[6], row[7]])
        except:
            pass

print(data_3[1])       
with open('aug_swap.txt', 'w') as f:
    for i in data_3:
        line = '{}'.format(i[0])+'\001'+i[1]+'\001'+i[2]+'\001'+'{}'.format(i[3])\
                +'\001'+i[4]+'\001'+i[5]+'\n'
        f.write(line)

with open('aug_sub.txt', 'w') as f:
    for i in data_2:
        line = '{}'.format(i[0])+'\001'+i[1]+'\001'+i[2]+'\001'+'{}'.format(i[3])\
                +'\001'+i[4]+'\001'+i[5]+'\n'
        f.write(line)

