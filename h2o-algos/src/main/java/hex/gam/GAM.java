package hex.gam;

import hex.DataInfo;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.gam.GAMModel.GAMParameters;
import hex.gam.MatrixFrameUtils.GamUtils;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMParameters;
import water.DKV;
import water.Key;
import water.Scope;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;

import java.util.Arrays;

import static hex.gam.MatrixFrameUtils.GamUtils.*;


public class GAM extends ModelBuilder<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ModelCategory.Regression};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Experimental;
  }

  @Override
  public boolean havePojo() {
    return false;
  }

  @Override
  public boolean haveMojo() {
    return false;
  }

  public GAM(boolean startup_once) {
    super(new GAMModel.GAMParameters(), startup_once);
  }

  public GAM(GAMModel.GAMParameters parms) {
    super(parms);
    init(false);
  }

  public GAM(GAMModel.GAMParameters parms, Key<GAMModel> key) {
    super(parms, key);
    init(false);
  }

/*  @Override
  public int nclasses() {
    if (_parms._family == GLMParameters.Family.multinomial || _parms._family == GLMParameters.Family.ordinal)
      return _nclass;
    if (_parms._family == GLMParameters.Family.binomial || _parms._family == GLMParameters.Family.quasibinomial)
      return 2;
    return 1;
  }*/
  
  @Override
  public void init(boolean expensive) {
    super.init(expensive);
    if (expensive) {  // add custom check here
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      
      if (_parms._gam_X == null)
        error("_gam_X", "must specify columns indices to apply GAM to.  If you don't have any, use GLM.");
      if (_parms._k == null) {  // user did not specify any knots, we will use default 10, evenly spread over whole range
        int numKnots = _train.numRows() < 10 ? (int) _train.numRows() : 10;
        _parms._k = new int[_parms._gam_X.length];  // different columns may have different 
        Arrays.fill(_parms._k, numKnots);
      }
      if ((_parms._saveGamCols || _parms._saveZMatrix) && ((_train.numCols() - 1 + _parms._k.length) < 2))
        error("_saveGamCols/_saveZMatrix", "can only be enabled if we number of predictors plus" +
                " Gam columns in _gamX exceeds 2");
      if ((_parms._lambda_search || !_parms._intercept || _parms._lambda == null || _parms._lambda[0] > 0)) 
        _parms._use_all_factor_levels = true;
    }
  }

  @Override
  protected boolean computePriorClassDistribution() {
    return (_parms._family== GLMParameters.Family.multinomial)||(_parms._family== GLMParameters.Family.ordinal);
  }
  
  @Override
  protected GAMDriver trainModelImpl() {
    return new GAMDriver();
  }

  @Override
  protected int nModelsInParallel(int folds) {
    return nModelsInParallel(folds, 2);
  }

  @Override
  protected void checkMemoryFootPrint_impl() {
    ;
  }

  private class GAMDriver extends Driver {
    boolean _centerGAM = false; // true if we need to constraint GAM columns
    double[][][] _zTranspose; // store for each GAM predictor transpose(Z) matrix
    double[][][] _penalty_mat;  // store for each GAM predictir the penalty matrix
    String[][] _gamColNames;  // store column names of GAM columns
    String[][] _gamColNamesCenter;  // gamColNames after de-centering is performed.
    Key<Frame>[] _gamFrameKeys;
    Key<Frame>[] _gamFrameKeysCenter;

    /***
     * This method will take the _train that contains the predictor columns and response columns only and add to it
     * the following:
     * 1. For each predictor included in gam_x, expand it out to calculate the f(x) and attach to the frame.
     * 2. It will calculate the ztranspose that is used to center the gam columns.
     * 3. It will calculate a penalty matrix used to control the smoothness of GAM.
     * 
     * @return
     */
    Frame adaptTrain() {
      int numGamFrame = _parms._gam_X.length;
      _centerGAM = (numGamFrame > 1) || (_train.numCols() - 1 + numGamFrame) >= 2;
      _zTranspose = _centerGAM ? GamUtils.allocate3DArray(numGamFrame, _parms, 0) : null;
      _penalty_mat = _centerGAM ? GamUtils.allocate3DArray(numGamFrame, _parms, 2) :
              GamUtils.allocate3DArray(numGamFrame, _parms, 1);
      _gamColNames = new String[numGamFrame][];
      _gamColNamesCenter = new String[numGamFrame][];
      _gamFrameKeys = new Key[numGamFrame];
      _gamFrameKeysCenter = new Key[numGamFrame];
      addGAM2Train(_parms, _parms.train(), _train, _zTranspose, _penalty_mat, _gamColNames, _gamColNamesCenter,
              true, _centerGAM, _gamFrameKeys, _gamFrameKeysCenter);
      return buildGamFrame(numGamFrame, _gamFrameKeys, _train, _parms._response_column);

/*
      Frame orig = _parms.train();  // contain all columns, _train contains only predictors and responses
      int numGamFrame = _parms._gam_X.length;
      Key<Frame>[] gamFramesKey = new Key[numGamFrame];  // store the Frame keys of generated GAM column
      RecursiveAction[] generateGamColumn = new RecursiveAction[numGamFrame];
      _centerGAM = (numGamFrame > 1) || (_train.numCols() - 1 + numGamFrame) >= 2; // only need to center when predictors > 1
      _zTranspose = _centerGAM ? GamUtils.allocate3DArray(numGamFrame, _parms, 0) : null;
      _penalty_mat = _centerGAM ? GamUtils.allocate3DArray(numGamFrame, _parms, 2) : GamUtils.allocate3DArray(numGamFrame, _parms, 1);
      for (int index = 0; index < numGamFrame; index++) {
        final Frame predictVec = new Frame(new String[]{_parms._gam_X[index]}, new Vec[]{orig.vec(_parms._gam_X[index])});  // extract the vector to work on
        final int numKnots = _parms._k[index];  // grab number of knots to generate
        final double scale = _parms._scale == null ? 1.0 : _parms._scale[index];
        final BSType splineType = _parms._bs[index];
        final int frameIndex = index;
        final String[] newColNames = new String[numKnots];
        for (int colIndex = 0; colIndex < numKnots; colIndex++) {
          newColNames[colIndex] = _parms._gam_X[index] + "_" + splineType.toString() + "_" + colIndex;
        }
        generateGamColumn[frameIndex] = new RecursiveAction() {
          @Override
          protected void compute() {
            GenerateGamMatrixOneColumn genOneGamCol = new GenerateGamMatrixOneColumn(splineType, numKnots, null, predictVec,
                    _parms._standardize, _centerGAM, scale).doAll(numKnots, Vec.T_NUM, predictVec);
            Frame oneAugmentedColumn = genOneGamCol.outputFrame(Key.make(), newColNames,
                    null);
            if (_centerGAM) {  // calculate z transpose
              oneAugmentedColumn = genOneGamCol.de_centralize_frame(oneAugmentedColumn,
                      predictVec.name(0) + "_" + splineType.toString() + "_decenter_", _parms);
              GamUtils.copy2DArray(genOneGamCol._ZTransp, _zTranspose[frameIndex]); // copy transpose(Z)
              double[][] transformedPenalty = ArrayUtils.multArrArr(ArrayUtils.multArrArr(genOneGamCol._ZTransp,
                      genOneGamCol._penaltyMat), ArrayUtils.transpose(genOneGamCol._ZTransp));  // transform penalty as zt*S*z
              GamUtils.copy2DArray(transformedPenalty, _penalty_mat[frameIndex]);
            } else
              GamUtils.copy2DArray(genOneGamCol._penaltyMat, _penalty_mat[frameIndex]); // copy penalty matrix
            gamFramesKey[frameIndex] = oneAugmentedColumn._key;
            DKV.put(oneAugmentedColumn);
          }
        };
      }
      ForkJoinTask.invokeAll(generateGamColumn);
      _gamColnames = new String[numGamFrame][];
      for (int frameInd = 0; frameInd < numGamFrame; frameInd++) {  // append the augmented columns to _train
        Frame gamFrame = gamFramesKey[frameInd].get();
        _gamColnames[frameInd] = new String[gamFrame.names().length];
        System.arraycopy(gamFrame.names(), 0, _gamColnames[frameInd], 0, gamFrame.names().length);
        _train.add(gamFrame.names(), gamFrame.removeAll());
        Scope.track(gamFrame);
      }
*/
    }

    @Override
    public void computeImpl() {
      init(true);     //this can change the seed if it was set to -1
      if (error_count() > 0)   // if something goes wrong, let's throw a fit
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);

      _job.update(0, "Initializing model training");

      buildModel(); // build gam model 
    }

    public final void buildModel() {
      GAMModel model = null;
      try {
        _job.update(0, "Adding GAM columns to training dataset...");
        Frame newTFrame = new Frame(_train.clone());
        Frame newTrain = new Frame(adaptTrain());
        DataInfo dinfo = new DataInfo(newTrain.clone(), _valid, 1, _parms._use_all_factor_levels || _parms._lambda_search, _parms._standardize ? DataInfo.TransformType.STANDARDIZE : DataInfo.TransformType.NONE, DataInfo.TransformType.NONE,
                _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.Skip,
                _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.MeanImputation || _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.PlugValues,
                _parms.makeImputer(),
                false, hasWeightCol(), hasOffsetCol(), hasFoldCol(), null);
        model = new GAMModel(dest(), _parms, new GAMModel.GAMModelOutput(GAM.this, dinfo._adaptedFrame));
        model.delete_and_lock(_job);
        newTFrame = _centerGAM?buildGamFrame(_parms._gam_X.length, _gamFrameKeysCenter, newTFrame, _parms._response_column):dinfo._adaptedFrame;  // get frames with correct predictors and spline functions
        DKV.put(newTFrame);
        if (_parms._saveGamCols) {
          DKV.put(newTrain);
          model._output._gamTransformedTrain =newTrain._key;  // access after model is built
          if (_centerGAM)
            model._output._gamTransformedTrainCenter = newTFrame._key;
        }
        _job.update(1, "calling GLM to build GAM model...");
        GLMModel glmModel = buildGLMModel(_parms, newTFrame); // obtained GLM model
        Scope.track_generic(glmModel);
        _job.update(0, "Building out GAM model...");
        fillOutGAMModel(glmModel, model, dinfo); // build up GAM model 
        
        // call adaptTeatForTrain() to massage frame before scoring.
        // call model.makeModelMetrics
        // create model summary by calling createModelSummaryTable or something like that
        model.update(_job);

        // build GAM Model Metrics
      } finally {
        if (model != null) 
          model.unlock(_job);
      }
    }
    
    GLMModel buildGLMModel(GAMParameters parms, Frame trainData) {
      GLMParameters glmParam = GamUtils.copyGAMParams2GLMParams(parms, trainData);  // copy parameter from GAM to GLM
      GLMModel model = new GLM(glmParam, _penalty_mat, _centerGAM? _gamColNamesCenter : _gamColNames).trainModel().get();
      return model;
    }
    
    void fillOutGAMModel(GLMModel glm, GAMModel model, DataInfo dinfo) {
      model._centerGAM = _centerGAM;
      model._gamColNames = _gamColNames;  // copy over gam column names
      model._gamColNamesCenter = _gamColNamesCenter;
      model._output._zTranspose = _zTranspose;
      model._gamFrameKeys = _gamFrameKeys;
      model._gamFrameKeysCenter = _gamFrameKeysCenter;
      if (_parms._savePenaltyMat)
        model._output._penaltyMatrices = _penalty_mat;
      copyGLMCoeffs(glm, model, dinfo);  // copy over coefficient names and generate coefficients as beta = z*GLM_beta
    }
    
    void copyGLMCoeffs(GLMModel glm, GAMModel model, DataInfo dinfo) {
      int totCoeffNums = dinfo.fullN()+1;
      model._output._coefficient_names = new String[totCoeffNums]; // copy coefficient names from GLM to GAM
      int gamNumStart = copyGLMCoeffNames2GAMCoeffNames(model, glm, dinfo);
      copyGLMCoeffs2GAMCoeffs(model, glm, dinfo, _parms._family, gamNumStart, _parms._standardize, _nclass); // obtain beta without centering
    }
  }
}
