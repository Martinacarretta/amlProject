# READ ME

## Files

### CSVs:
- [merged_sports](merged_sports.csv) has the paths, and superclass of all the images 
- [superclass_merged_sports](superclass_merged_sports.csv) has the paths, superclasses and subclasses of all images

- [train](train.csv) is used to then get the csv without a desired % of unlabeled data. It has 70% of the overall data

- [unlabelled_train_10pct](unlabelled_train_10pct.csv) is used to train with a 10% of unlabeled images
- [unlabelled_train_20pct](unlabelled_train_20pct.csv) is used to train with a 20% of unlabeled images
- [unlabelled_train_50pct](unlabelled_train_50pct.csv) is used to train with a 50% of unlabeled images
- [unlabelled_train_70pct](unlabelled_train_70pct.csv) is used to train with a 70% of unlabeled images
- [unlabelled_train_100pct](unlabelled_train_100pct.csv) is used to train with a 100% of unlabeled images

- [val](val.csv) is used to perform validation. It has 15% of the overall data
- [test](test.csv) is used to test, it has about 15% of the overall data

### Train:
Use [train_test_split](<train_test split.ipynb>) to generate the desired split csvs. 
Use [train](train.py) as the main python script.
Use [tgpu](tgpu.sh) to send the job. 

### Test:
Use the [best](hierarchical_sports_model_20pct_penal0p8.pth) model to test on unseen data. 
### Others:
[train_backup](train_backup.py) has the previous train where the superclass was high but the subclass was not. NEEDS MORE EPOCHS