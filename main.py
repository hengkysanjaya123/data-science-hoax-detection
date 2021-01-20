import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from sklearn.linear_model import PassiveAggressiveClassifier
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from text_preprocessing import stem
from sklearn import svm
import timeit


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

def hoax_detection():
    #Reading data as pandas dataframe
    frame = pd.read_csv('MasterBeritaAfterCleanCombined.csv', error_bad_lines=False, encoding='latin1')
    frame2 = pd.read_csv('new_TestData.csv', error_bad_lines=False, encoding='latin1')
    
    # TODO: remove this line
#    frame = frame.head(5)
    
    berita = ''
    berita = stem(berita)
    data = {
                'no' : ['1'],
                'berita' : [berita],
                'tagging': ['Hoax']
        }
#    frame2 = pd.DataFrame(data, columns=['no','berita','tagging'])
    
    #Inspecing Shape
    frame.shape
    frame2.shape
    
    #Inspecting top 5 rows
    frame.head()
    frame2.head()
    
    #Setting the DataFrame index (row labels) using one or more existing columns
    frame = frame.set_index("no")
    frame.head()
    
    frame2 = frame2.set_index("no")
    frame2.head()
    
    y = frame.tagging
    y.head()
    
    y2 = frame2.tagging
    
    frame.drop("tagging", axis=1)
    frame.head()
    
    frame2.drop("tagging", axis=1)
    
    # print(frame['berita'])
    
    
    
    # print(frame['berita'])    
    
    X_train = frame['berita']
    y_train = y
    print(X_train.shape)
    print(y_train.shape)
    # print(X_train)
    # print(y_train)
    # print(len(X_train))
    # print(len(y_train))
    
    # uux_train, X_test , uuy_train, y_test = train_test_split(frame2['berita'], y2, test_size=0.33, random_state=53)
    
    X_test = frame2['berita']
    y_test = y2
    print(len(X_test))
    
    # stemming
    # print(frame['berita'][0])
    
    # print(frame2['berita'])
    
    X_train.head()
    
    y_train.head()
    
    X_train, X_test, y_train, y_test = train_test_split(frame['berita'], y, test_size=0.33, random_state=53)
    
    
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    
 
       
    # count_vectorizer = case folding, tokenizing, remove stopwords
    # analyze = count_vectorizer.build_analyzer()
    # analyze("Saya mau MAKAN dimakan di tempat makan")
    # print(count_vectorizer)
    # count_vectorizer = CountVectorizer(lowercase=True, stop_words=frozenset(stopwords))

    # Fit and transform the training data.
    # count_train = count_vectorizer.fit_transform(X_train)
    
    # print(count_train)
    # Transform the test set 
    # count_test = count_vectorizer.transform(X_test)
    
    
    # Initialize the `tfidf_vectorizer` 
    tfidf_vectorizer = TfidfVectorizer(lowercase=True,stop_words=frozenset(stopwords), max_df=0.7)
    
    # Fit and transform the training data 
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    
    # Transform the test set 
    tfidf_test = tfidf_vectorizer.transform(X_test)
    
    
    print(tfidf_test)
    
    print('separator')
    
    # Get the feature names of `tfidf_vectorizer` 
    print(tfidf_vectorizer.get_feature_names()[-20:])
    
    
    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
    
#    tfidf_df.to_excel('output-hoax-only.xlsx')
    
#    print(tfidf_df)
    # Get the feature names of `count_vectorizer` 
    # print(count_vectorizer.get_feature_names()[0:10])
    
    
    import matplotlib.pyplot as plt
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        See full source and example: 
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    
    start = timeit.default_timer()

    clf = MultinomialNB() 
    clf.fit(tfidf_train, y_train)
    pred = clf.predict(tfidf_test)
    score = accuracy_score(y_test, pred)
    multinomialpred = pred
    print("#Result:#Multinomial#", pred)
    print("accuracy:   %0.3f" % score)
    cm = confusion_matrix(y_test, pred, labels=['Hoax', 'Valid'])
    stop = timeit.default_timer()
    print('Time Multinomial: ', stop - start)  
    plot_confusion_matrix(cm, classes=['Hoax', 'Valid'], title='MultinomialNB Confusion Matrix (Predict: Test)')
    
    
#    y_pred_prob = clf.predict_proba(tfidf_test)
#    print(y_pred_prob)
#    hoax_probs = y_pred_prob[:,1]
#
#    
#    fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=hoax_probs, pos_label='spam')
#    # Plot
#    plt.plot(fpr,tpr, color='red')
#    plt.title('Receiver Operating Characteristic Curve', size=20)
#    plt.plot([0, 1], [0, 1], color='green', linestyle=':')
#    plt.xlabel('False Positive Rate', size=15)
#    plt.ylabel('True Positive Rate', size=15)
#    plt.show()
    
    clf = MultinomialNB() 
    clf.fit(tfidf_train, y_train)
    pred = clf.predict(tfidf_train)
    score = accuracy_score(y_train, pred)
    multinomialpred = pred
    cm = confusion_matrix(y_train, pred, labels=['Hoax', 'Valid'])
    stop = timeit.default_timer()
    plot_confusion_matrix(cm, classes=['Hoax', 'Valid'], title='MultinomialNB Confusion Matrix (Predict: Training)')
    
    
    start = timeit.default_timer()
    linear_clf = PassiveAggressiveClassifier()
    linear_clf.fit(tfidf_train, y_train)
    pred = linear_clf.predict(tfidf_test)
    score = accuracy_score(y_test, pred)
    passiveaggressivepred = pred
    print("#Result:#PassiveAggressiveClassifier#", pred)
    print("accuracy:   %0.3f" % score)
    cm = confusion_matrix(y_test, pred, labels=['Hoax', 'Valid'])
    stop = timeit.default_timer()
    print('Time PassiveAggressiveClassifier: ', stop - start) 
    plot_confusion_matrix(cm, classes=['Hoax', 'Valid'], title='PassiveAggressiveClassifier Confusion Matrix (Predict: Test)')
    
    linear_clf = PassiveAggressiveClassifier()
    linear_clf.fit(tfidf_train, y_train)
    pred = linear_clf.predict(tfidf_train)
    score = accuracy_score(y_train, pred)
    passiveaggressivepred = pred
    cm = confusion_matrix(y_train, pred, labels=['Hoax', 'Valid'])
    stop = timeit.default_timer()
    print('Time PassiveAggressiveClassifier: ', stop - start) 
    plot_confusion_matrix(cm, classes=['Hoax', 'Valid'], title='PassiveAggressiveClassifier Confusion Matrix (Predict: Training)')
    
    
    start = timeit.default_timer()
    linear_clf_svm = svm.SVC()
    linear_clf_svm.fit(tfidf_train, y_train)
    pred = linear_clf_svm.predict(tfidf_test)
    score = accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    print("#Result:#SVM#", pred)    
    svmpred = pred
    cm = confusion_matrix(y_test, pred, labels=['Hoax', 'Valid'])
    stop = timeit.default_timer()
    print('Time SVM: ', stop - start) 
    plot_confusion_matrix(cm, classes=['Hoax', 'Valid'],title='SVM Confusion Matrix (Predict: Test)')
    
    linear_clf_svm = svm.SVC()
    linear_clf_svm.fit(tfidf_train, y_train)
    pred = linear_clf_svm.predict(tfidf_train)
    score = accuracy_score(y_train, pred) 
    svmpred = pred
    cm = confusion_matrix(y_train, pred, labels=['Hoax', 'Valid'])
    stop = timeit.default_timer()
    print('Time SVM: ', stop - start) 
    plot_confusion_matrix(cm, classes=['Hoax', 'Valid'],title='SVM Confusion Matrix (Predict: Training)')
    
    
    def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
        """
        See: https://stackoverflow.com/a/26980472
        
        Identify most important features if given a vectorizer and binary classifier. Set n to the number
        of weighted features you would like to show. (Note: current implementation merely prints and does not 
        return top classes.)
        """
    
        class_labels = classifier.classes_
        feature_names = vectorizer.get_feature_names()
        topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
        topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]
    
        for coef, feat in topn_class1:
            print(class_labels[0], coef, feat)
    
        print()
    
        for coef, feat in reversed(topn_class2):
            print(class_labels[1], coef, feat)
    

    print('y_test')
    print(y_test)
    
    
#    print('score')
#    print(score)
    
#    y_pred_prob = clf.predict_proba(tfidf_test)
#    spam_probs = y_pred_prob[:,1]
#    print(spam_probs)
#    
#    # Build confusion metrics
#    fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=spam_probs, pos_label='spam')
#    # Plot
#    plt.plot(fpr,tpr, color='red')
#    plt.title('Receiver Operating Characteristic Curve', size=20)
#    plt.plot([0, 1], [0, 1], color='green', linestyle=':')
#    plt.xlabel('False Positive Rate', size=15)
#    plt.ylabel('True Positive Rate', size=15)
#    plt.show()

    
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, linear_clf.decision_function(tfidf_test), pos_label='neg')
    # find threshold closest to zero:
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, 
     label='threshold zero(default)', fillstyle='none', c='k', mew=2)
    plt.plot([0,1], linestyle='-', lw=2, color='r', label='random', alpha=0.8)
    plt.legend(loc=4)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (recall)')
    plt.title('roc_curve');
    plt.show()
    from sklearn.metrics import auc
    print('AUC score is: ', auc(fpr, tpr));
    
    
    
        
    # plot precision recall curve Multinomial
#    disp = plot_precision_recall_curve(linear_clf, tfidf_test, y_test)
#    y_score = linear_clf.decision_function(X_test)
#    average_precision = average_precision_score(y_test, y_score)
#    disp.ax_.set_title('2-class Precision-Recall curve: '
#                       'AP={0:0.2f}'.format(average_precision))
#    disp.show()
#    
#    most_informative_feature_for_binary_classification(tfidf_vectorizer, linear_clf, n=30)
    
    feature_names = tfidf_vectorizer.get_feature_names()
    sorted(zip(clf.coef_[0], feature_names), reverse=True)[:20]
    
    ### Most fake
    sorted(zip(clf.coef_[0], feature_names))[:20]
    
    tokens_with_weights = sorted(list(zip(feature_names, clf.coef_[0])))
    for i in tokens_with_weights:
        print(i)
        break
    
    result = dict();
    result['multinomial'] = multinomialpred
    result['passive'] = passiveaggressivepred
    result['svm'] = svmpred
    
    # print(result)
    return result
    

hoax_detection()
#hoax_detection('Jakarta, Di jejaring sosial, banyak beredar informasi yang menyebut lele sebagaiikan paling jorok. Dalam sesuap daging ikan lele, terkandung 3000 sel kanker. Benarkah?Julukan sebagai ikan paling jorok merujuk pada sifat lele yang doyan mengonsumsisegala jenis limbah di perairan. Bahkan sebuah artikel yang cukup viral di internetmenyebutkan kotoran manusia juga dijadikan pakan pada sebuah budidaya lele di Kota Haikou, China.Sementara itu di habitat aslinya, lele atau catfish juga dikenal sebagai spesies ikan yangsangat tangguh. Ikan ini dilengkapi alat pernapasan tambahan berupa labirin,sehingga mampu bertahan hidup dalam kondisi perairan berlumpur atau bahkan tercemar.Agaknya, fakta inilah yang memunculkan dugaan soal akumulasi racun karsinogen (penyebab kanker) di tubuh ikan lele.Untungnya, ikan lele yang beredar di pasaran bukan berasal dari alam liar.Lele banyak dibudidayakan di kolam-kolam, yang mestinya bisa dikendalikan agar bebas dari pencemaran.Pakan yang diberikan juga bisa dipilih, tidak harus mengandalkan limbah.Yang pasti, popularitas ikan bersungut ini tidak pernah pudar, bahkan terus meningkat. Data Kementerian Kelautan dan Perikanan (KKP) menyebut produksi lele pada 2013 mencapai 543,461 ton, meningkat dari 441,217 ton pada 2012 dan 337,577 ton pada 2011.Konsumsi ikan lele menurut Badan Pusat Statistik (BPS) tercatat 29,98 kg / kapita / tahun, naik dari 22,58 kg / kapita / tahun pada 2004. Di Jakarta, tak kurang dari 6000 lapak pecel lele telah terdaftar di Asosiasi Pedagang Kaki Lima Indonesia (APKLI).Soal kandungan nutrisi, tak bisa dipungkiri bahwa lele adalah sumber protein berharga yang murah meriah. Fakta bahwa ikan lele juga rendah kolesterol sepertinya bakal menenggelamkan tudingan bahwa lele bisa memicu kanker. """"Saat ini belum ada penelitian yang menyatakan jika memakan lele dapat memicu kanker,"""" tegas dr Dradjat R Suardi, SpB(K)Onk, ahli kanker dari Perhimpunan Onkologi Indonesia saat dihubungi detikHealth, Jumat (23/10/2015).""')