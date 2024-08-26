import pandas as pd, numpy as np, re

from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
import matplotlib.pyplot as plt

#######################################################################################################
nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#######################################################################################################
    
root = tk.Tk()
root.title("Spam EMail Detection SYSTEM")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

image2 =Image.open(r'img3.jpg')
image2 =image2.resize((w,h))\
    
    
    

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)

###########################################################################################################
lbl = tk.Label(root, text="___SPAM EMAIL DETECTION SYSTEM___", font=('times', 35,' bold '), height=1, width=60,bg="#FFBF40",fg="black")
lbl.place(x=0, y=0)
##############################################################################################################################



##############################################################################################################


def all_ana():
    
   
    
    image2 =Image.open(r'bar_graph.png')
    image2 =image2.resize((400,400), Image.ANTIALIAS)

    background_image=ImageTk.PhotoImage(image2)

    background_label = tk.Label(root, image=background_image)
    background_label.image = background_image

    background_label.place(x=850, y=150)

def SVM():
    
    result = pd.read_csv("spam.csv",encoding = 'unicode_escape')

    result.head()
        
    result['Message_without_stopwords'] = result['Message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
 ###########################################################################################################################################
    
    def pos(email_without_stopwords):
        return TextBlob(email_without_stopwords).tags
    
    
    os = result.Message_without_stopwords.apply(pos)
    os1 = pd.DataFrame(os)
    #
    os1.head()
    
    os1['pos'] = os1['Message_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))
    
    result = result = pd.merge(result, os1, right_index=True, left_index=True)
    result.head()
    result['pos']
    email_train, email_test, label_train, label_test = train_test_split(result['pos'], result['Category'],
                                                                              test_size=0.2,random_state=8)
    
    tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    X_train_tf = tf_vect.fit_transform(email_train)
    X_test_tf = tf_vect.transform(email_test)
    
    
    
    clf = svm.SVC(C=10, kernel='linear', gamma=0.001,random_state=123)   
    clf.fit(X_train_tf, label_train)
    pred = clf.predict(X_test_tf)
    
    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(tf_vect, fin)
    with open('mlmodel.pickle', 'wb') as f:
        pickle.dump(clf, f)
    
    pkl = open('mlmodel.pickle', 'rb')
    clf = pickle.load(pkl)
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    
    X_test_tf = tf_vect.transform(email_test)
    pred = clf.predict(X_test_tf)
    
    print(metrics.accuracy_score(label_test, pred))
    
    print(confusion_matrix(label_test, pred))
    
    print(classification_report(label_test, pred))

       
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(label_test, pred)))
    print("Accuracy : ",accuracy_score(label_test, pred)*100)
    accuracy = accuracy_score(label_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(label_test, pred) * 100)
    repo = (classification_report(label_test, pred))
    
    label4 = tk.Label(root,text =str(repo),width=50,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=300,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as SVM_MODEL.joblib",width=50,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=300,y=300)
    
    dump (clf,"SVM_MODEL.joblib")
    print("Model saved as SVM_MODEL.joblib")
    
    
def DT():
    
    result = pd.read_csv("spam.csv",encoding = 'unicode_escape')

    result.head()
        
    result['Message_without_stopwords'] = result['Message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
 ###########################################################################################################################################
    
    def pos(email_without_stopwords):
        return TextBlob(email_without_stopwords).tags
    
    
    os = result.Message_without_stopwords.apply(pos)
    os1 = pd.DataFrame(os)
    #
    os1.head()
    
    os1['pos'] = os1['Message_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))
    
    result = result = pd.merge(result, os1, right_index=True, left_index=True)
    result.head()
    result['pos']
    email_train, email_test, label_train, label_test = train_test_split(result['pos'], result['Category'],
                                                                              test_size=0.2,random_state=8)
    
    tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    X_train_tf = tf_vect.fit_transform(email_train)
    X_test_tf = tf_vect.transform(email_test)
    
    
    ###########################################################################################################################
    
    
    from sklearn.tree import DecisionTreeClassifier
    
    
    clf =  DecisionTreeClassifier()  
    clf.fit(X_train_tf, label_train)
    pred = clf.predict(X_test_tf)
    
    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(tf_vect, fin)
    with open('mlmodel.pickle', 'wb') as f:
        pickle.dump(clf, f)
    
    pkl = open('mlmodel.pickle', 'rb')
    clf = pickle.load(pkl)
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    
    X_test_tf = tf_vect.transform(email_test)
    pred = clf.predict(X_test_tf)
    
    print(metrics.accuracy_score(label_test, pred))
    
    print(confusion_matrix(label_test, pred))
    
    print(classification_report(label_test, pred))

       
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(label_test, pred)))
    print("Accuracy : ",accuracy_score(label_test, pred)*100)
    accuracy = accuracy_score(label_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(label_test, pred) * 100)
    repo = (classification_report(label_test, pred))
    
    label4 = tk.Label(root,text =str(repo),width=50,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=300,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as DT_MODEL.joblib",width=50,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=300,y=300)
    
    dump (clf,"DT_MODEL.joblib")
    print("Model saved as DT_MODEL.joblib")
    

def NB():
    result = pd.read_csv("spam.csv", encoding='unicode_escape')
    result['Message_without_stopwords'] = result['Message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    def pos(email_without_stopwords):
        return TextBlob(email_without_stopwords).tags

    os = result.Message_without_stopwords.apply(pos)
    os1 = pd.DataFrame(os)
    os1['pos'] = os1['Message_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))

    result = pd.merge(result, os1, right_index=True, left_index=True)

    email_train, email_test, label_train, label_test = train_test_split(result['pos'], result['Category'],
                                                                        test_size=0.2, random_state=8)

    tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)

    X_train_tf = tf_vect.fit_transform(email_train)
    X_test_tf = tf_vect.transform(email_test)
    
    # Convert sparse matrices to dense arrays
    X_train_tf_dense = X_train_tf.toarray()
    X_test_tf_dense = X_test_tf.toarray()
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(X_train_tf_dense, label_train)
    pred = clf.predict(X_test_tf_dense)
    
    print("Accuracy: %.2f%%" % (accuracy_score(label_test, pred) * 100.0))
    print(confusion_matrix(label_test, pred))
    print(classification_report(label_test, pred))
    
    # Save model and vectorizer
    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(tf_vect, fin)
    with open('mlmodel.pickle', 'wb') as f:
        pickle.dump(clf, f)

    # Load model and vectorizer
    pkl = open('mlmodel.pickle', 'rb')
    clf = pickle.load(pkl)
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)

    # Convert test data to tf-idf representation
    X_test_tf = tf_vect.transform(email_test)
    X_test_tf_dense = X_test_tf.toarray()

    # Predict using loaded model
    pred = clf.predict(X_test_tf_dense)

    print(metrics.accuracy_score(label_test, pred))
    
    print(confusion_matrix(label_test, pred))
    
    print(classification_report(label_test, pred))

       
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(label_test, pred)))
    print("Accuracy : ",accuracy_score(label_test, pred)*100)
    accuracy = accuracy_score(label_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(label_test, pred) * 100)
    repo = (classification_report(label_test, pred))
    
    label4 = tk.Label(root,text =str(repo),width=50,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=300,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as NB_MODEL.joblib",width=50,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=300,y=300)
    
    dump (clf,"NB_MODEL.joblib")
    print("Model saved as NB_MODEL.joblib")
    

################################################################################################################################################################

frame = tk.LabelFrame(root,text="Control Panel",width=250,height=600,bd=3,background="teal",font=("Tempus Sanc ITC",15,"bold"))
frame.place(x=15,y=100)

Entry_frame = tk.LabelFrame(root,text="Enter Input",width=500,height=600,bd=3,background="teal",font=("Tempus Sanc ITC",15,"bold"))
Entry_frame.place(x=850,y=100)

entry = tk.Entry(Entry_frame,width=30,font=("Times new roman",15,"bold"))
entry.insert(0,"")
entry.place(x=100,y=60)
##############################################################################################################################################################################
def Test():
    predictor = load("SVM_MODEL.joblib")
    Given_text = entry.get()
    #Given_text = "the 'roseanne' revival catches up to our thorny po..."
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    X_test_tf = tf_vect.transform([Given_text])
    y_predict = predictor.predict(X_test_tf)
    print(y_predict[0])
    if y_predict[0]=="ham":
        label4 = tk.Label(Entry_frame,text ="Mail is ham: \nHam messages are \nthe intended or safe \nlegitimate messages in a mailbox",width=25,height=5,bg='green',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=10,y=250)
    elif y_predict[0]=="spam":
        label4 = tk.Label(Entry_frame,text ="Mail is spam: \nSpam messages are \nthe junk, unsolicited bulk or \ncommercial messages \nin the mailbox.",width=25,height=5,bg='red',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=10,y=250)
    
    
###########################################################################################################################################################

def window():
    root.destroy()

def display():
        from subprocess import call
        call(["python","display.py"])

# button2 = tk.Button(frame,command=Train,text="Train",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
# button2.place(x=25,y=100)





button3 = tk.Button(frame,command=SVM,text="SVM",bg="#26619c",fg="black",width=15,font=("Times New Roman",15,"bold"))
button3.place(x=25,y=100)

button3 = tk.Button(frame,command=DT,text="DT",bg="#26619c",fg="black",width=15,font=("Times New Roman",15,"bold"))
button3.place(x=25,y=200)

button2 = tk.Button(frame,command=NB,text="NB",bg="#26619c",fg="black",width=15,font=("Times New Roman",15,"bold"))
button2.place(x=25,y=300)


button3 = tk.Button(Entry_frame,command=Test,text="Test",bg="#26619c",fg="black",width=15,font=("Times New Roman",15,"bold"))
button3.place(x=160,y=150)

button3 = tk.Button(frame,command=display,text="Data Display",bg="#26619c",fg="black",width=15,font=("Times New Roman",15,"bold"))
button3.place(x=25,y=400)

button4 = tk.Button(frame,command=window,text="Exit",bg="red",fg="black",width=15,font=("Times New Roman",15,"bold"))
button4.place(x=25,y=500)




root.mainloop()