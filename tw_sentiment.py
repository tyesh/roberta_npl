from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
 
tweet = "Wasn't there no evidence? You bastard @HLGatell that's what you are fucking doctor death."
 
tweet_words = []
 
for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    elif word.startswith('http'):
        word = 'http'
    tweet_words.append(word)
 
tweet_proc = " ".join(tweet_words)
print(tweet_proc)
 
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
 
model = AutoModelForSequenceClassification.from_pretrained(roberta)
 
tokenizer = AutoTokenizer.from_pretrained(roberta)
 
labels = ['Negative', 'Neutral', 'Positive']
 
encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
 
output = model(**encoded_tweet)
 
scores = output[0][0].detach().numpy()
 
scores = softmax(scores)
 
for i in range(len(scores)):
    l = labels[i]
    s = scores[i]
    print(l, s)
