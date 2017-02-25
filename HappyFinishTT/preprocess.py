import os
import numpy as np
import tokenisation_process as tp

def remove_all(text, stripWords):
    for word in stripWords:
        text=text.replace(word, '')
    return text

def preprocess(path_to_csv = 'amazon_book_reviews/',test=False):
    scores = []
    addresses = []
    titles = []
    reviews = []
    i=0
    for file in os.listdir(path_to_csv):
        if file[-3:] == 'csv':
            path = os.path.join(path_to_csv,file)

            f = open(path)

            reviewStripWords = ['\n','\r','<span class=""a-size-base review-text"">', '</span>',  '<br/>']
            linesList = f.readlines()
            for line in linesList:

                score,address, title, review = line.split('\t')
                scores.append(float(score))
                addresses.append(address)
                titles.append(title)
                reviewStripped = remove_all(review, reviewStripWords)
                reviewStripped = reviewStripped.replace('.',' ')
                reviewStripped = reviewStripped.replace(',',' ')
                reviewStripped = reviewStripped.replace(';',' ')
                reviews.append(reviewStripped)



                i+=1


    np.save('review',np.asarray(reviews))
    np.save('scores',np.asarray(scores))
    
    tp.main(np.asarray(reviews),np.asarray(scores),test=test)
