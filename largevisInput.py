
import time; start_time = time.time()
#import numpy as np
import pandas as pd
#import fileinput
#import sys


train = pd.read_csv('numerai_training_data.csv', encoding="utf-8-sig")
f=open('nai.train.rows','w')
train_rows=train.shape[0]
f.write(str(train_rows))
f.close()
 
test = pd.read_csv('numerai_tournament_data.csv', encoding="utf-8-sig")
f=open('nai.test.rows','w')
test_rows=test.shape[0] 
f.write(str(test_rows))
f.close()



train = train.drop(['target'],axis=1)

test_id = test['t_id']
test_id.to_csv("numerai_tournament_data_id.csv",index=False, header=True)

test = test.drop(['t_id'],axis=1)


df_all = pd.concat((train, test), axis=0, ignore_index=True)
rows = df_all.shape[0]
columns = df_all.shape[1]
#print(rows, columns)

df_all_copy = df_all
header=str(list(df_all_copy.columns.values))# with []
f=open('nai.header','w')
f.write(header)
f.close()

df_all.to_csv("nai.txt",index=False, header=False, sep=' ')

# (rows,columns)


################### not safte
# open the file again for writing
f =open('nai.txt')
nai = f.read()
f.close()

f = open('nai.txt', 'w')
f.write(str(rows)+" "+str(columns)+"\n")
f.write(nai)
f.close()
###################

