<a name="readme-top"></a>

# Array phase detection using TPhaseNet models

Code related to the submitted paper **Adapting deep learning phase detectors for seismic array processing**.

This is work in progress and will be completed until the paper is published.

Dummy training data from ARCES, SPITS and FINES arrays are provided. Complete test data will be uploaded here:

https://doi.org/10.17605/OSF.IO/27FPK

Tested setup for installation of required packages  :

```
conda create -n test python=3.10.12
conda activate test
pip3 install -r requirements.txt

```

Edit config.yaml to choose type of input data, type of model, etc.
Several model config files are also provided.

## Single station detection

Train model on single station data (explanation of output files will be added):

```
python train.py --config config_1stat.yaml

```

Create evaluation metrics and plots on test data:

```
python evaluate_on_testdata.py --config config_1stat.yaml

```

## Ensemble detection

Apply single station model to all array stations in test data for ensemble detection:


```
python predict_on_testdata.py --config config_1statfullarray.yaml

```

Combining single station output for ensemble detection is done in evaluation script:

```
python evaluate_on_testdata.py --config config_1statfullarray.yaml
```

## Beam detection:

Train two beam detection models (vertical ad three-component beams) and evaulate on test data:

```
python train.py --config config_zbeam.yaml
python train.py --config config_3cbeam.yaml
python evaluate_on_testdata.py --config config_zbeam.yaml
python evaluate_on_testdata.py --config config_3cbeam.yaml
```

## Array detection:

Train the array detection model (ARCES array using Set 2 - see paper) and evaluate on test data:

```
python train.py --config config_arrayarces_set2.yaml
python evaluate_on_testdata.py --config config_arrayarces_set2.yaml
```

## Detection on contineous data

### Single station detection (on ARA0):

In config_1stat.yaml set:

prediction.predict : True # run on contineous data

prediction.combine_array_stations : False # no post-processing (only done for ensemble and beam detection)

prediction.detect_only : False # If True will not write out continoues detection time series

prediction.stations: ['ARA0']


```
python predict_contineous.py --config config_1stat.yaml
```

### Ensemble detection:

Do the detection on all array elements with single station detection model:

Im model_configs/config_1stat.yaml set :

prediction.predict : True # run on contineous data

prediction.combine_array_stations : ['stack'] # do post-processing (stacking)

prediction.stations: ['ARCES'] # use all ARCES stations

```
python predict_contineous.py --config config_1stat.yaml
```

Do not write detection time series for each station, but combine and write only phase detectiobs (only works for stacking option):

prediction.detect_only : True

To be added:
- evaluation of contineous processing results
- contineous processing using beam detection
- contineous processing using array detection
