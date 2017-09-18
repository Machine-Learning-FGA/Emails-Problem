from make_dict import make_dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from termcolor import cprint 
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.lancaster import LancasterStemmer
from sklearn.metrics import confusion_matrix

#directory files
pasta = make_dict()

#data and targe
x=[]
y=[]

directory = '20news_fewheader/'

lt = LancasterStemmer()

#return a list os word
def words_of_file(path):
	with open(path,encoding='ISO-8859-1') as f:
		words = f.read().split()
#	for i in range(len(words)):
#		words[i]=lt.stem(words[i])
	return words


for key in pasta.keys():
	print("Correndo por: (%s)" %key,end='')
	for files in pasta[key]:
		val=words_of_file(directory+key+'/'+files)
		x+=val
		for k in range(len(val)):
			y.append(key)
	cprint(' Done', 'green')

print(len(x))
print(len(y))

cprint( 'YES' if len(x)==len(y) else 'NO', 'green')

#salvando dados de treino 

print('Salvando dados... ', end='')
with open('outx.txt','w') as output:
	output.write(str(x))
cprint('Complete', 'yellow')


#salvando targets 

print('Salvando target... ', end='')
with open('outy.txt', 'w') as output:
	output.write(str(y))
cprint('Complete', 'yellow')


print('Iniciando Feacture Selection')
cv = CountVectorizer()
result_x = cv.fit_transform(x)
#x_train, x_test, y_train, y_test = train_test_split(result_x,y,test_size=0.4,random_state=42)

nb = MultinomialNB()
nb.fit(result_x,y)
cv=cross_val_score(nb,result_x,y)
print(cv)

cprint("Fit train", 'yellow') 



cprint("END!", 'green')



