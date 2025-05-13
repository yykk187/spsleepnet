# SPSleep
### SPSleepNet: Enhancing EEG-Based Sleep Staging for OSA Patients via Sleep Position Integration


## Requirmenets:
- Intall jq package (for linux)
- Python3.12
- Pytorch=='2.6.0'
- Numpy=='2.2.1'
- Sklearn
- Pandas
- openpyxl
- mne=='1.9.0'

## Prepare datasets

- [ISRUC of Subgroup_1 ](https://sleeptight.isr.uc.pt/)

After downloading the datasets, you can prepare them as follows:
```
cd prepare_datasets
python prepare_ISRUC.py --data_dir  --output_dir  --select_ch
```

## Training SPSleep

```
python main.py 
```
## Results
The log file of each fold is found in the fold directory inside the results.
