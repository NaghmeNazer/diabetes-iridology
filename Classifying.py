import sklearn.svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def mlp_classify(x_train, y_train, x_test, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(15, 10), max_iter=2000,activation="logistic")

    scores_acc          = cross_val_score(mlp, x_train, y_train, cv=5, scoring='accuracy')
    scores_recall       = cross_val_score(mlp, x_train, y_train, cv=5, scoring='recall')
    scores_presision    = cross_val_score(mlp, x_train, y_train, cv=5, scoring='average_precision')

    print "MLP: ",  scores_acc.mean(), scores_recall.mean(), scores_presision.mean()


    return scores_acc.mean(), scores_recall.mean(), scores_presision.mean()


def rf_classify(x_train, y_train, x_test, y_test):
    rf = RandomForestClassifier(n_estimators=7)  # initialize

    scores_acc = cross_val_score(rf, x_train, y_train, cv=5, scoring='accuracy')
    scores_recall = cross_val_score(rf, x_train, y_train, cv=5, scoring='recall')
    scores_presision = cross_val_score(rf, x_train, y_train, cv=5, scoring='average_precision')

    print "RF: ", scores_acc.mean(), scores_recall.mean(), scores_presision.mean()

    return scores_acc.mean(), scores_recall.mean(), scores_presision.mean()


def linearsvm_classify(x_train, y_train, x_test, y_test):
    kernelType = 'linear'
    model = sklearn.svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
                            decision_function_shape='ovo', gamma='auto', kernel=kernelType, degree=5,
                            max_iter=-1, probability=True, random_state=None, shrinking=True,
                            tol=0.001, verbose=False)

    scores_acc = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    scores_recall = cross_val_score(model, x_train, y_train, cv=5, scoring='recall')
    scores_presision = cross_val_score(model, x_train, y_train, cv=5, scoring='average_precision')

    print "LSVM: ", scores_acc.mean(), scores_recall.mean(), scores_presision.mean()


    return scores_acc.mean(), scores_recall.mean(), scores_presision.mean()


def adaboost_classify(x_train, y_train, x_test, y_test):
    bdt = AdaBoostClassifier(algorithm="SAMME", n_estimators=30)

    scores_acc = cross_val_score(bdt, x_train, y_train, cv=5, scoring='accuracy')
    scores_recall = cross_val_score(bdt, x_train, y_train, cv=5, scoring='recall')
    scores_presision = cross_val_score(bdt, x_train, y_train, cv=5, scoring='average_precision')

    print "Adaboost: ", scores_acc.mean(), scores_recall.mean(), scores_presision.mean()

    return scores_acc.mean(), scores_recall.mean(), scores_presision.mean()

def ensemble_classify_hardVoting(x_train, y_train, x_test, y_test):
    mlp     = MLPClassifier(hidden_layer_sizes=(15, ), max_iter=2000, alpha=1e-4,
                        solver='sgd', random_state=1, learning_rate_init=.1, activation='logistic',
                        learning_rate='adaptive')
    rf      = RandomForestClassifier(n_estimators=25)  # initialize
    lsvm    = sklearn.svm.SVC(C=0.01, cache_size=100, class_weight=None, coef0=0.0,
                            decision_function_shape='ovo', gamma='auto', kernel='linear', degree=3,
                            max_iter=-1, probability=True, random_state=None, shrinking=True,
                            tol=0.001, verbose=False)
    neighbors = KNeighborsClassifier(n_neighbors=10, p=2, algorithm='auto')
    ada       = AdaBoostClassifier(algorithm="SAMME", n_estimators=25)

    clf = [lsvm,ada,mlp]
    eclf = EnsembleVoteClassifier(clfs=clf, voting='hard')

    scores_acc = cross_val_score(eclf, x_train, y_train, cv=5, scoring='accuracy')
    scores_recall = cross_val_score(eclf, x_train, y_train, cv=5, scoring='recall')
    scores_presision = cross_val_score(eclf, x_train, y_train, cv=5, scoring='average_precision')

    print "ENS: ", scores_acc.mean(), scores_recall.mean(), scores_presision.mean()

    return scores_acc.mean(), scores_recall.mean(), scores_presision.mean()


def lda_classify(X, y, parameters, score, k_fold_num, shrinkage=None):
    if shrinkage == None:
        print("# Tuning hyper-parameters for %s" % score)
        clf = GridSearchCV(LinearDiscriminantAnalysis(), parameters, cv=k_fold_num, scoring=score)
    else:
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)

    scores_acc = cross_val_score(clf, X, y, cv=k_fold_num, scoring='accuracy')
    scores_recall = cross_val_score(clf, X, y, cv=k_fold_num, scoring='recall')
    scores_presision = cross_val_score(clf, X, y, cv=k_fold_num, scoring='average_precision')
    print "LDA: ", scores_acc.mean(), scores_recall.mean(), scores_presision.mean()

    return scores_acc.mean(), scores_recall.mean(), scores_presision.mean()




def lda_classify2(X_train, y_train, X_test, y_test, parameters, score, k_fold_num, shrinkage=None):

    if shrinkage == None:
        print("# Tuning hyper-parameters for %s" % score)
        clf = GridSearchCV(LinearDiscriminantAnalysis(), parameters, cv=k_fold_num, scoring=score)
    else:
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)

    clf.fit(X_train, y_train)

    if shrinkage == None:
        print("Best parameters set found on development set:")
        print(clf.best_params_)

    y_true, y_pred = y_test, clf.predict(X_test)
    return y_true, y_pred