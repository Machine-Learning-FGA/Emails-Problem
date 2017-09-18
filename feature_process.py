import pandas as pd
from termcolor import cprint
from sklearn.model_selection import train_test_split

def clean_list(path):
	with open(path,'r') as f:
		x = f.read().split(',')

	del x[0]
	del x[-1]

	x= [str(i).replace("\'","") for i in x]
	return x


print("Importanto lista de dados...", end='')
X = clean_list('outx.txt')
cprint('Done!', 'green')


print("Importanto lista de dados...", end='')
Y = clean_list('outy.txt')

cprint('Done!', 'green')


from sklearn.feature_extraction.text import CountVectorizer
print('Iniciando Feacture Selection')
cv = CountVectorizer(stop_words='english')

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4)

cprint("Fit train", 'yellow')

result_x = cv.fit_transform(x_test)

print(result_x.size)

cprint("END!", 'green')

