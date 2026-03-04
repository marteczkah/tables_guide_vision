<div align="center">

  <h1><a href="https://arxiv.org/pdf/2503.14998">🫀👀 Tables Guide Vision: Learning to See the Heart through Tabular Data (WACV2026)</a></h1>

</div>

This is the official repository of the WACV2026 paper _Tables Guide Vision: Learning to See the Heart through Tabular Data_. Make sure to cite our paper if this code was useful.

```bibtex
@inproceedings{hasny2026tables,
  title={Tables Guide Vision: Learning to See the Heart Through Tabular Data},
  author={Hasny, Marta and Di Folco, Maxime and Bressem, Keno and Schnabel, Julia},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1716--1725},
  year={2026}
}
```

## Instructions

### Data
Our codebase provides the code to run TGV with twa datasets that were used in the paper, UK Biobank and DVM. UK Biobank is semi-private dataset, you can apply for access <a href='https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/'>here</a>. DVM is open-access and can be found under this <a href='https://deepvisualmarketing.github.io'>link</a>. We provide the code for preprocessing of DVM in ```data_prep/prep_dvm.ipynb```. The code for preprocessing UKBB will be uploaded soon, meanwhile we provide the code used to generate h5 files that we used for training given csv files with the tabular data and image paths. 

### Pretraining
To run the pretraining using TGV use script ```train_tgv.py```, which supports pretraining using both DVM (use flag ```-d 2```) and UKBB (use flag ```-d 3```). A sample command including best hyperparameters from the paper for UKBB would be:
```
python train_tgv.py -p train.h5 -v val.h5 -b 512 -e 10 -t 0.1 -s 'tgv_ukbb' -l 1e-3 -h 0.05 --augment 0.0 -d 3
```

And for DVM:
```
python train_tgv.py -p train_paths_all_views.pt --tabular_train_path dvm_features_train_noOH_all_views_physical_labeled.csv -v val_paths_all_views.pt --tabular_val_path dvm_features_train_noOH_all_views_physical_labeled.csv -b 512 -e 500 -t 0.1 -s 'tgv_dvm' -l 1e-4 -h 0.1 --augment 0.95 -d 2
```
The models will be saved in the folder specified under the flag ```-s```.

### Fine-tuning
The pretrained models can be applied on classification and regression tasks. UKBB supports multilabel classification (```train_multilabel.py```) and regression ```train_regression.py -d 3```), while DVM supports multiclass classification ```train_multiclass.py```) and regression ```train_regression.py -d 2```).

### Zero-shot Experiments
The code for our zero-shot experiments (no need for further training) will be available soon. 

## Checkpoints
We provide checkpoints for our pretrained models. The UKBB checkpoint will be released upon approval.

Dataset | DVM | Cardiac
--- | --- | ---
Checkpoints | [Download](https://drive.google.com/file/d/1qU1eZV7xHNSPK-nGhXakKy5MHn867POj/view?usp=sharing) | Pending approval.

## Acknowledgments
We thank those repositories for their great work:
- [MMCL](https://github.com/paulhager/MMCL-Tabular-Imaging)
