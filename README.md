# Just_Stack_Them_Up
A python implementation to add models using the stacking methodology. Highly influenced by a post from MLWave
http://mlwave.com/kaggle-ensembling-guide/

# Usage

base_classifiers = [[RandomForestClassifier(n_estimators= 2, criterion = 'entropy'),"RF_ENTROPY"],                                                          [RandomForestClassifier(n_estimators = 2, criterion = 'gini'),"RF_GINI"]],

blender_Classifiers =[[RandomForestClassifier(n_estimators = 2, criterion = 'entropy'),"BLEND_RF_ENTROPY"],                                                          [RandomForestClassifier(n_estimators = 2, criterion = 'gini'),"BLEND_RF_GINI"]]

#Create a stacker obj

<pre><code>
stack1 = stacker(x_test = X_test,x_train = X_train,y_train =y_train,id_test=id_test, base_clf_list=base_classifiers,blender_clf_list =                   blender_Classifiers)
</pre></code>
Here the id_test is the ID column of a test dataset that will be required.

You can train all base classifiers and create blended_train and test dataset using:

<pre><code>
stack1.train_all_base_classifiers()
</pre></code>

#OUTPUT
<p>
Training classifier [0] [RF_ENTROPY]
Fold [0]
auc_score for fold: 0.601556465779
Fold [1]
auc_score for fold: 0.604232041248
Fold [2]
auc_score for fold: 0.584394983439
Fold [3]
auc_score for fold: 0.603993230476
Fold [4]
auc_score for fold: 0.59159561713
cv_score_mean: 0.597154467614 and cv_score_std: 0.00787329116686
Training classifier [1] [RF_GINI]
Fold [0]
auc_score for fold: 0.590317422147
Fold [1]
auc_score for fold: 0.604126251307
Fold [2]
auc_score for fold: 0.598296676694
Fold [3]
auc_score for fold: 0.593156671037
Fold [4]
auc_score for fold: 0.579752596971
cv_score_mean: 0.593129923631 and cv_score_std: 0.00817897680863
</p>

# Train All blenders using:

<pre><code>
stack1.train_all_blenders()
</pre></code>

# Find Cross Validation AUC score of all Blenders using:

<pre><code>
stack1.find_cv_scores_all_blenders()
</pre></code>

<p>
blender_Name: BLEND_RF_ENTROPY :
Fold 1 CV Score: 0.602361151841
Fold 2 CV Score: 0.625628341057
Fold 3 CV Score: 0.596159573553
Fold 4 CV Score: 0.601513229218
Fold 5 CV Score: 0.596663814886
cv_score_mean: 0.604465222111 and cv_score_std: 0.0108707380607
blender_Name: BLEND_RF_GINI :
Fold 1 CV Score: 0.597490833245
Fold 2 CV Score: 0.614910113814
Fold 3 CV Score: 0.595923008772
Fold 4 CV Score: 0.604254460162
Fold 5 CV Score: 0.591763066213
cv_score_mean: 0.600868296441 and cv_score_std: 0.00809205888706
</p>

# Print the object at anytime to see whats up with it.

<pre><code>
print(stack1)
</pre></code>

<p>
init:
n_folds: 5 random_seed:0
base classifiers:
1. [RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=2, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False), 'RF_ENTROPY'] 
fold_1_auc:0.601556465779
fold_2_auc:0.604232041248
fold_3_auc:0.584394983439
fold_4_auc:0.603993230476
fold_5_auc:0.59159561713
2. [RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=2, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False), 'RF_GINI'] 
fold_1_auc:0.590317422147
fold_2_auc:0.604126251307
fold_3_auc:0.598296676694
fold_4_auc:0.593156671037
fold_5_auc:0.579752596971
Blender classifiers:
1. [RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=2, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False), 'BLEND_RF_ENTROPY'] 
fold_1_auc:0.602361151841
fold_2_auc:0.625628341057
fold_3_auc:0.596159573553
fold_4_auc:0.601513229218
fold_5_auc:0.596663814886
2. [RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=2, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False), 'BLEND_RF_GINI'] 
fold_1_auc:0.597490833245
fold_2_auc:0.614910113814
fold_3_auc:0.595923008772
fold_4_auc:0.604254460162
fold_5_auc:0.591763066213
</ps>

