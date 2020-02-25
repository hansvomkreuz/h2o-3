import sys
sys.path.insert(1,"../../")

from tests import pyunit_utils
import h2o, tests
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.xgboost import *
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.isolation_forest import H2OIsolationForestEstimator
from h2o.estimators.aggregator import H2OAggregatorEstimator
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import numpy as np


def test_set_auto_parameters_to_default_values():
    #distribution (available in Deep Learning, XGBoost, GBM):
    #stopping_metric (available in: GBM, DRF, Deep Learning, AutoML, XGBoost, Isolation Forest):
    #histogram_type (Available in: GBM, DRF)
    #solver (available in: GLM) already done in hex.glm.GLM.defaultSolver()
    #categorical_encoding (Available in: GBM, DRF, Deep Learning, K-Means, Aggregator, XGBoost, Isolation Forest)
    #fold_assignment (Available in: GBM, DRF, Deep Learning, GLM, Naïve-Bayes, K-Means, XGBoost)
    #todo glm test, naive bayes test
    
    #Deep Learning:
    # Training data
    train_data = h2o.import_file(path=tests.locate("smalldata/gbm_test/ecology_model.csv"))
    train_data = train_data.drop('Site')
    train_data['Angaus'] = train_data['Angaus'].asfactor()
    print(train_data.describe())
    train_data.head()

    # Testing data
    test_data = h2o.import_file(path=tests.locate("smalldata/gbm_test/ecology_eval.csv"))
    test_data['Angaus'] = test_data['Angaus'].asfactor()
    print(test_data.describe())
    test_data.head()

    # Run DeepLearning
    dl1 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True, stopping_rounds=5)
    dl1.train(x=list(range(1,train_data.ncol)),
             y="Angaus",
             training_frame=train_data,
             validation_frame=test_data)

    dl2 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True, distribution="bernoulli", categorical_encoding="OneHotInternal", stopping_rounds=5)
    dl2.train(x=list(range(1,train_data.ncol)),
              y="Angaus",
              training_frame=train_data,
              validation_frame=test_data)

    assert dl1.actual_params['distribution'] == dl2.actual_params['distribution']
    assert dl1.logloss() == dl2.logloss()
    assert dl1.actual_params['stopping_metric'] == dl2.actual_params['stopping_metric']
    assert dl1.actual_params['categorical_encoding'] == dl2.actual_params['categorical_encoding']
    assert dl1.actual_params['fold_assignment'] == dl2.actual_params['fold_assignment']

    dl1 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True, nfolds=5)
    dl1.train(x=list(range(1,train_data.ncol)),
              y="Angaus",
              training_frame=train_data,
              validation_frame=test_data)

    dl2 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True, distribution="bernoulli", categorical_encoding="OneHotInternal", nfolds=5, fold_assignment="Random")
    dl2.train(x=list(range(1,train_data.ncol)),
              y="Angaus",
              training_frame=train_data,
              validation_frame=test_data)

    assert dl1.actual_params['distribution'] == dl2.actual_params['distribution']
    assert dl1.logloss() == dl2.logloss()
    assert dl1.actual_params['stopping_metric'] == dl2.actual_params['stopping_metric']
    assert dl1.actual_params['categorical_encoding'] == dl2.actual_params['categorical_encoding']
    assert dl1.actual_params['fold_assignment'] == dl2.actual_params['fold_assignment']



    #XGBoost:
    assert H2OXGBoostEstimator.available()

    prostate_frame = h2o.import_file(pyunit_utils.locate('smalldata/prostate/prostate.csv'))
    x = ['RACE']
    y = 'CAPSULE'
    prostate_frame[y] = prostate_frame[y].asfactor()

    prostate_frame.split_frame(ratios=[0.75], destination_frames=['prostate_training', 'prostate_validation'], seed=1)

    training_frame = h2o.get_frame('prostate_training')
    test_frame = h2o.get_frame('prostate_validation')

    # Model with maximum of 2 trees
    model_2_trees1 = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7,
                                         booster='gbtree', seed=1, ntrees=2, stopping_rounds =5)
    model_2_trees1.train(x=x, y=y, training_frame=training_frame)

    # Model with maximum of 2 trees
    model_2_trees2 = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7,
                                         booster='gbtree', seed=1, ntrees=2, distribution="bernoulli", categorical_encoding="OneHotInternal", stopping_rounds =5 )
    model_2_trees2.train(x=x, y=y, training_frame=training_frame)

    assert model_2_trees1.actual_params['distribution'] == model_2_trees2.actual_params['distribution']
    assert model_2_trees1.logloss() == model_2_trees2.logloss()
    assert model_2_trees1.actual_params['stopping_metric'] == model_2_trees2.actual_params['stopping_metric']
    assert model_2_trees1.actual_params['categorical_encoding'] == model_2_trees2.actual_params['categorical_encoding']
    assert model_2_trees1.actual_params['fold_assignment'] == model_2_trees2.actual_params['fold_assignment']


    # Model with maximum of 2 trees
    model_2_trees1 = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7,
                                        booster='gbtree', seed=1, ntrees=2, nfolds=5)
    model_2_trees1.train(x=x, y=y, training_frame=training_frame)

    # Model with maximum of 2 trees
    model_2_trees2 = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7,
                                         booster='gbtree', seed=1, ntrees=2, distribution="bernoulli", categorical_encoding="OneHotInternal", nfolds=5, fold_assignment="Random")
    model_2_trees2.train(x=x, y=y, training_frame=training_frame)

    assert model_2_trees1.actual_params['distribution'] == model_2_trees2.actual_params['distribution']
    assert model_2_trees1.logloss() == model_2_trees2.logloss()
    assert model_2_trees1.actual_params['stopping_metric'] == model_2_trees2.actual_params['stopping_metric']
    assert model_2_trees1.actual_params['categorical_encoding'] == model_2_trees2.actual_params['categorical_encoding']
    assert model_2_trees1.actual_params['fold_assignment'] == model_2_trees2.actual_params['fold_assignment']

    #GBM:
    cars = h2o.import_file(path=pyunit_utils.locate("smalldata/junit/cars_20mpg.csv"))
    
    cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()
    cars["year"] = cars["year"].asfactor()
    
    predictors = ["displacement","power","weight","acceleration","year"]
    response = "economy_20mpg"

    train, valid = cars.split_frame(ratios = [.8], seed = 1234)

    cars_gbm1 = h2o.estimators.gbm.H2OGradientBoostingEstimator(seed = 1234, stopping_rounds = 3)
    cars_gbm1.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

    cars_gbm2 = h2o.estimators.gbm.H2OGradientBoostingEstimator(seed = 1234, stopping_rounds = 3, distribution="bernoulli", stopping_metric="logloss", histogram_type="UniformAdaptive", categorical_encoding="Enum")
    cars_gbm2.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

    assert cars_gbm1.logloss() == cars_gbm2.logloss()
    assert cars_gbm1.actual_params['distribution'] == cars_gbm2.actual_params['distribution']
    assert cars_gbm1.actual_params['stopping_metric'] == cars_gbm2.actual_params['stopping_metric']
    assert cars_gbm1.actual_params['histogram_type'] == cars_gbm2.actual_params['histogram_type']
    assert cars_gbm1.actual_params['stopping_metric'] == cars_gbm2.actual_params['stopping_metric']
    assert cars_gbm1.actual_params['categorical_encoding'] == cars_gbm2.actual_params['categorical_encoding']
    print("GBM:categorical_encoding:")
    print(cars_gbm1.actual_params['categorical_encoding'])
    
    #GBM fold assignment:

    cars_gbm1 = h2o.estimators.gbm.H2OGradientBoostingEstimator(seed = 1234, nfolds=5)
    cars_gbm1.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

    cars_gbm2 = h2o.estimators.gbm.H2OGradientBoostingEstimator(seed = 1234, nfolds=5, fold_assignment='Random', distribution="bernoulli", histogram_type="UniformAdaptive", categorical_encoding="Enum")
    cars_gbm2.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

    assert cars_gbm1.logloss() == cars_gbm2.logloss()
    assert cars_gbm1.actual_params['distribution'] == cars_gbm2.actual_params['distribution']
    assert cars_gbm1.actual_params['stopping_metric'] == cars_gbm2.actual_params['stopping_metric']
    assert cars_gbm1.actual_params['histogram_type'] == cars_gbm2.actual_params['histogram_type']
    assert cars_gbm1.actual_params['fold_assignment'] == cars_gbm2.actual_params['fold_assignment']
    assert cars_gbm1.actual_params['stopping_metric'] == cars_gbm2.actual_params['stopping_metric']
    assert cars_gbm1.actual_params['categorical_encoding'] == cars_gbm2.actual_params['categorical_encoding']

    #KMEANS
    km_h2o1 = H2OKMeansEstimator(seed=1234, categorical_encoding="AUTO", nfolds=5)
    km_h2o1.train(x=["economy_20mpg","displacement","power","weight","acceleration","year"], training_frame=cars)
    print("KMEANS:categorical_encoding:")
    print(km_h2o1.actual_params['categorical_encoding'])

    km_h2o2 = H2OKMeansEstimator(seed=1234, categorical_encoding="Enum", nfolds=5, fold_assignment='Random')
    km_h2o2.train(x=["economy_20mpg","displacement","power","weight","acceleration","year"], training_frame=cars)
    print("KMEANS:categorical_encoding:")
    print(km_h2o2.actual_params['categorical_encoding'])
    assert km_h2o1.actual_params['categorical_encoding'] == km_h2o2.actual_params['categorical_encoding']
    assert km_h2o1.actual_params['fold_assignment'] == km_h2o2.actual_params['fold_assignment']
   
    #GBM:not categorical
    df = h2o.import_file(path=pyunit_utils.locate("smalldata/logreg/prostate.csv"))
    df.describe()
    # Remove ID from training frame
    df.pop('ID')  # WARNING: this is NOT python negative indexing!!!
    # For VOL & GLEASON, a zero really means "missing"
    df[df['VOL'],'VOL'] = None
    df[df['GLEASON'],'GLEASON'] = None
    # Test/train split
    r = df.runif()
    train = df[r < 0.8]
    test  = df[r >= 0.8]
    # Run GBM
    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    gbm = H2OGradientBoostingEstimator(ntrees=5, max_depth=3)
    gbm.train(x=list(range(2,train.ncol)), y="CAPSULE", training_frame=train, validation_frame=test)
    assert gbm.actual_params['categorical_encoding'] == 'None'
   
   #DRF

    df = h2o.import_file(path=pyunit_utils.locate("smalldata/gbm_test/ecology_model.csv"))
    df["Angaus"] = df["Angaus"].asfactor()
    df["Weights"] = h2o.H2OFrame.from_python(abs(np.random.randn(df.nrow, 1)).tolist())[0]
    print(df.col_names)
    train, calib = df.split_frame(ratios=[.8], destination_frames=["eco_train", "eco_calib"], seed=42)

    rf1 = H2ORandomForestEstimator(
        ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5,
        weights_column="Weights", stopping_rounds = 3,
        calibrate_model=True, calibration_frame=calib, seed = 1234
    )
    rf1.train(
        x=list(range(2, train.ncol)),
        y="Angaus", training_frame=train
    )

    rf2 = H2ORandomForestEstimator(
        ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5,
        weights_column="Weights",stopping_rounds = 3, stopping_metric='logloss',
        calibrate_model=True, calibration_frame=calib, seed = 1234, categorical_encoding = 'Enum'
    )
    rf2.train(
        x=list(range(2, train.ncol)),
        y="Angaus", training_frame=train
    )

    assert rf1.actual_params['stopping_metric'] == rf2.actual_params['stopping_metric']
    assert rf1.logloss() == rf2.logloss()
    assert rf1.actual_params['distribution'] == rf2.actual_params['distribution']
    assert rf1.actual_params['categorical_encoding'] == 'Enum'

    rf1 = H2ORandomForestEstimator(
        ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5,
        weights_column="Weights", nfolds = 5,
        calibrate_model=True, calibration_frame=calib, seed = 1234
    )
    rf1.train(
        x=list(range(2, train.ncol)),
        y="Angaus", training_frame=train
    )

    rf2 = H2ORandomForestEstimator(
        ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5,
        weights_column="Weights", nfolds = 5, fold_assignment='Random',
        calibrate_model=True, calibration_frame=calib, seed = 1234, categorical_encoding = 'Enum'
    )
    rf2.train(
        x=list(range(2, train.ncol)),
        y="Angaus", training_frame=train
    )

    assert rf1.actual_params['stopping_metric'] == rf2.actual_params['stopping_metric']
    assert rf1.logloss() == rf2.logloss()
    assert rf1.actual_params['distribution'] == rf2.actual_params['distribution']
    assert rf1.actual_params['fold_assignment'] == 'Random'
    assert rf1.actual_params['categorical_encoding'] == 'Enum'

    print("DRF:categorical_encoding:")
    print(rf1.actual_params['categorical_encoding'])
    
    #ISOLATION FORREST
    train = h2o.import_file(pyunit_utils.locate("smalldata/anomaly/ecg_discord_train.csv"))

    if_model1 = H2OIsolationForestEstimator(ntrees=7, seed=12, sample_size=5, stopping_rounds=3)
    if_model1.train(training_frame=train)

    if_model2 = H2OIsolationForestEstimator(ntrees=7, seed=12, sample_size=5, stopping_rounds=3, stopping_metric = 'anomaly_score' )
    if_model2.train(training_frame=train)

    assert if_model1.actual_params['stopping_metric'] == if_model2.actual_params['stopping_metric']
    assert if_model1._model_json['output']['training_metrics']._metric_json['mean_score'] == if_model2._model_json['output']['training_metrics']._metric_json['mean_score']
    assert if_model1.actual_params['categorical_encoding'] == if_model2.actual_params['categorical_encoding']

    print("ISOLATION FOREST:categorical_encoding:")
    print(if_model1.actual_params['categorical_encoding'])

    #categorical IF:
    if_cat_model1 = H2OIsolationForestEstimator(ntrees=7, seed=12, sample_size=5, stopping_rounds=3)
    if_cat_model1.train(training_frame=cars)
    print("ISOLATION FOREST CATEGORICAL:categorical_encoding:")
    print(if_cat_model1.actual_params['categorical_encoding'])


    #AGGREGATOR
    df = h2o.create_frame(
        rows=10000,
        cols=10,
        categorical_fraction=0.6,
        integer_fraction=0,
        binary_fraction=0,
        real_range=100,
        integer_range=100,
        missing_fraction=0,
        factors=100,
        seed=1234
    )
    agg1 = H2OAggregatorEstimator(target_num_exemplars = 1000, rel_tol_num_exemplars = 0.5, categorical_encoding = "eigen")
    agg1.train(training_frame=df)  
    agg2 = H2OAggregatorEstimator(target_num_exemplars = 1000, rel_tol_num_exemplars = 0.5)
    agg2.train(training_frame=df)
    assert agg1.actual_params['categorical_encoding'] == agg2.actual_params['categorical_encoding']




if __name__ == "__main__":
    pyunit_utils.standalone_test(test_set_auto_parameters_to_default_values)
else:
    test_set_auto_parameters_to_default_values()
