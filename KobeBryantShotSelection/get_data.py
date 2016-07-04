# __author__ = 'weij'
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier
import utils

sns.set_style('whitegrid')
pd.set_option('display.max_columns', None) # display all columns
data = pd.read_csv('input/data.csv')

data.set_index('shot_id', inplace=True)
data["action_type"] = data["action_type"].astype('object')
data["combined_shot_type"] = data["combined_shot_type"].astype('category')
data["game_event_id"] = data["game_event_id"].astype('category')
data["game_id"] = data["game_id"].astype('category')
data["period"] = data["period"].astype('object')
data["playoffs"] = data["playoffs"].astype('category')
data["season"] = data["season"].astype('category')
data["shot_made_flag"] = data["shot_made_flag"].astype('category')
data["shot_type"] = data["shot_type"].astype('category')
data["team_id"] = data["team_id"].astype('category')
# print data.head(2)
# print data.dtypes
# print data.shape
# print data.describe(include=['number'])
# print data.describe(include=['object', 'category'])
# csv1 = data.describe(include=['object', 'category'])
# csv1.to_csv('output/col_stat.csv')
# ax = plt.axes()
# sns.countplot(x='shot_made_flag', data=data, ax=ax);
# ax.set_title('Shooting Average')
# plt.show()
# utils.get_acc(data, 'shot_distance')
# data['game_date_DT'] = pd.to_datetime(data['game_date'])
# data['dayOfWeek'] = data['game_date_DT'].dt.dayofweek
# data['dayOfYear'] = data['game_date_DT'].dt.dayofyear
#
# data['secondsFromPeriodEnd'] = 60*data['minutes_remaining']+data['seconds_remaining']
# data['secondsFromPeriodStart'] = 60*(11-data['minutes_remaining'])+(60-data['seconds_remaining'])
# data['secondsFromGameStart'] = (data['period'] <= 4).astype(int)*(data['period']-1)*12*60 + (data['period'] > 4).astype(int)*((data['period']-4)*5*60 + 3*12*60) + data['secondsFromPeriodStart']

# look at first couple of rows and verify that everything is good
# data.ix[:20,['period','minutes_remaining','seconds_remaining','secondsFromGameStart']]

# utils.timeshot(data)
# utils.draw_pic(data)

########################################################################################
data_cl = data.copy() # create a copy of data frame
target = data_cl['shot_made_flag'].copy()

# Remove some columns
data_cl.drop('team_id', axis=1, inplace=True) # Always one number
data_cl.drop('lat', axis=1, inplace=True) # Correlated with loc_x
data_cl.drop('lon', axis=1, inplace=True) # Correlated with loc_y
data_cl.drop('game_id', axis=1, inplace=True) # Independent
data_cl.drop('game_event_id', axis=1, inplace=True) # Independent
data_cl.drop('team_name', axis=1, inplace=True) # Always LA Lakers
data_cl.drop('shot_made_flag', axis=1, inplace=True)
def detect_outliers(series, whis=1.5):
    q75, q25 = np.percentile(series, [75 ,25])
    iqr = q75 - q25
    return ~((series - series.median()).abs() <= (whis * iqr))
# Remaining time
data_cl['seconds_from_period_end'] = 60 * data_cl['minutes_remaining'] + data_cl['seconds_remaining']
data_cl['last_5_sec_in_period'] = data_cl['seconds_from_period_end'] < 5

data_cl.drop('minutes_remaining', axis=1, inplace=True)
data_cl.drop('seconds_remaining', axis=1, inplace=True)
data_cl.drop('seconds_from_period_end', axis=1, inplace=True)

## Matchup - (away/home)
data_cl['home_play'] = data_cl['matchup'].str.contains('vs').astype('int')
data_cl.drop('matchup', axis=1, inplace=True)

# Game date
data_cl['game_date'] = pd.to_datetime(data_cl['game_date'])
data_cl['game_year'] = data_cl['game_date'].dt.year
data_cl['game_month'] = data_cl['game_date'].dt.month
data_cl.drop('game_date', axis=1, inplace=True)

# Loc_x, and loc_y binning
data_cl['loc_x'] = pd.cut(data_cl['loc_x'], 25)
data_cl['loc_y'] = pd.cut(data_cl['loc_y'], 25)

# Replace 20 least common action types with value 'Other'
rare_action_types = data_cl['action_type'].value_counts().sort_values().index.values[:20]
data_cl.loc[data_cl['action_type'].isin(rare_action_types), 'action_type'] = 'Other'
categorial_cols = [
    'action_type', 'combined_shot_type', 'period', 'season', 'shot_type',
    'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_year',
    'game_month', 'opponent', 'loc_x', 'loc_y']

for cc in categorial_cols:
    dummies = pd.get_dummies(data_cl[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    data_cl.drop(cc, axis=1, inplace=True)
    data_cl = data_cl.join(dummies)
unknown_mask = data['shot_made_flag'].isnull()
# Separate dataset for validation
data_submit = data_cl[unknown_mask]

# Separate dataset for training
X = data_cl[~unknown_mask]
Y = target[~unknown_mask]

before_X = X.copy()
before_Y = Y.copy()


threshold = 0.90
vt = VarianceThreshold().fit(X)

# Find feature names
feat_var_threshold = data_cl.columns[vt.variances_ > threshold * (1-threshold)]
print feat_var_threshold
model = RandomForestClassifier()
model.fit(X, Y)

feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index
# print feat_imp_20

X_minmax = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
X_scored = SelectKBest(score_func=chi2, k='all').fit(X_minmax, Y)
feature_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': X_scored.scores_
    })

feat_scored_20 = feature_scoring.sort_values('score', ascending=False).head(20)['feature'].values
# feat_scored_20

rfe = RFE(LogisticRegression(), 20)
rfe.fit(X, Y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })

feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
# feat_rfe_20

features = np.hstack([
        feat_var_threshold,
        feat_imp_20,
        feat_scored_20,
        feat_rfe_20
    ])

features = np.unique(features)
print('Final features set:\n')
# for f in features:
#     print("\t-{}".format(f))

#######################################################################################
data_cl = data_cl.ix[:, features]
data_submit = data_submit.ix[:, features]
X = X.ix[:, features]
print 'X=',X.shape[1]
print('Clean dataset shape: {}'.format(data_cl.shape))
print('Subbmitable dataset shape: {}'.format(data_submit.shape))
print('Train features shape: {}'.format(X.shape))
print('Target label shape: {}'. format(Y.shape))

# components = 8
# pca = PCA(n_components=components).fit(X)
#
# pca_variance_explained_df = pd.DataFrame({
#     "component": np.arange(1, components+1),
#     "variance_explained": pca.explained_variance_ratio_
#     })
#
# ax = sns.barplot(x='component', y='variance_explained', data=pca_variance_explained_df)
# ax.set_title("PCA - Variance explained")
# plt.show()
import datetime
def process_alg(_X, _Y):
    seed = 7
    processors=1
    num_folds=5
    num_instances=len(_X)
    scoring='log_loss'
    # scoring='accuracy'

    kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    # Prepare some basic models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('RFC', RandomForestClassifier()))
    models.append(('K-NN', KNeighborsClassifier(n_neighbors=5)))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    # models.append(('SVC', SVC(probability=True)))

    # Evaluate each model in turn
    results = []
    names = []

    for name, model in models:
        startime = datetime.datetime.now()
        cv_results = cross_val_score(model, _X, _Y, cv=kfold, scoring=scoring, n_jobs=processors)
        results.append(cv_results)
        names.append(name)
        endtime = datetime.datetime.now()
        timecost = (endtime-startime).seconds
        print("{0}: {1:.3f} +- {2:.3f}\t {3}s".format(name, cv_results.mean(), cv_results.std() ,timecost))

from sklearn.metrics import roc_curve, auc
def draw_ROC(_X,_Y,index):
    X_train, X_test, y_train, y_test = train_test_split(_X, _Y, test_size=.5,
                                                    random_state=0)
    # n_classes = _Y.shape[1]
    n_classes = 1
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('RFC', RandomForestClassifier()))
    # models.append(('K-NN', KNeighborsClassifier(n_neighbors=10)))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    models.append(('SVC', SVC()))
    plt.figure(index)
    for name, model in models:
        y_score = model.fit(X_train, y_train).decision_function(X_test)
        print y_score
        print y_test
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # for i in range(n_classes):
        #     print i
        i=0
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr[i], tpr[i], label='{0}: (area = {1:0.2f})'.format(name, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



print 'before'
# print before_X.head()
# print before_Y.head()
# print before_X.dtypes
process_alg(before_X,before_Y)
# draw_ROC(before_X,before_Y,1)
print 'after'
# draw_ROC(X,Y,2)
process_alg(X,Y)
# print X.head()
# print Y.head()
# print X.dtypes

print before_X.shape[1], X.shape[1]

seed = 7
processors=1
num_folds=5
num_instances=len(X)
scoring='log_loss'
    # scoring='accuracy'
kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)

# gbm_grid = GridSearchCV(
#     estimator = GradientBoostingClassifier(warm_start=True, random_state=seed),
#     param_grid = {
#         'n_estimators': [100, 200],
#         'max_depth': [2, 3, 4],
#         'max_features': [10, 15, 20],
#         'learning_rate': [1e-1, 1]
#     },
#     cv = kfold,
#     scoring = scoring,
#     n_jobs = processors)
#
# gbm_grid.fit(X, Y)
#
# print(gbm_grid.best_score_)
# print(gbm_grid.best_params_)
#
# lr_grid = GridSearchCV(
#     estimator = LogisticRegression(random_state=seed),
#     param_grid = {
#         'penalty': ['l1', 'l2'],
#         'C': [0.001, 0.01, 1, 10, 100, 1000]
#     },
#     cv = kfold,
#     scoring = scoring,
#     n_jobs = processors)
#
# lr_grid.fit(X, Y)
#
# print(lr_grid.best_score_)
# print(lr_grid.best_params_)
#
#
# lda_grid = GridSearchCV(
#     estimator = LinearDiscriminantAnalysis(),
#     param_grid = {
#         'solver': ['lsqr'],
#         'shrinkage': [0, 0.25, 0.5, 0.75, 1],
#         'n_components': [None, 2, 5, 10]
#     },
#     cv = kfold,
#     scoring = scoring,
#     n_jobs = processors)
#
# lda_grid.fit(X, Y)
#
# print(lda_grid.best_score_)
# print(lda_grid.best_params_)
#
#
# rf_grid = GridSearchCV(
#     estimator = RandomForestClassifier(warm_start=True, random_state=seed),
#     param_grid = {
#         'n_estimators': [100, 200],
#         'criterion': ['gini', 'entropy'],
#         'max_features': [18, 20],
#         'max_depth': [8, 10],
#         'bootstrap': [True]
#     },
#     cv = kfold,
#     scoring = scoring,
#     n_jobs = processors)
#
# rf_grid.fit(X, Y)
#
# print(rf_grid.best_score_)
# print(rf_grid.best_params_)

estimators = []

estimators.append(('lr', LogisticRegression(penalty='l2', C=1)))
estimators.append(('lda', LinearDiscriminantAnalysis(shrinkage=0, n_components=None, solver='lsqr')))
estimators.append(('gbm', GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, max_features=15, warm_start=True, random_state=seed)))
estimators.append(('rf', RandomForestClassifier(bootstrap=True, max_depth=8, n_estimators=200, max_features=20, criterion='entropy', random_state=seed)))

# create the ensemble model
ensemble = VotingClassifier(estimators, voting='soft', weights=[2,3,3,1])

results = cross_val_score(ensemble, X, Y, cv=kfold, scoring=scoring,n_jobs=processors)
print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
