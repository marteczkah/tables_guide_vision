<div align="center">

  <h1><a href="https://arxiv.org/pdf/2503.14998">🫀👀 Tables Guide Vision: Learning to See the Heart through Tabular Data (WACV2026)</a></h1>

</div>

This is the official repository of the WACV2026 paper _Tables Guide Vision: Learning to See the Heart through Tabular Data_. Make sure to cite our paper if this code was useful.

```bibtex
@inproceedings{hasny2026tables,
  title={Tables guide vision: Learning to see the heart through tabular data},
  author={Hasny, Marta and Di Folco, Maxime and Bressem, Keno and Schnabel, Julia},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1716--1725},
  year={2026}
}
```

## Instructions

### Data
Our codebase provides the code to run TGV with twa datasets that were used in the paper, UK Biobank and DVM. UK Biobank is semi-private dataset, you can apply for access <a href='https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/'>here</a>. DVM is open-access and can be found under this <a href='https://deepvisualmarketing.github.io'>link</a>. We provide the code for preprocessing of DVM in ```data_prep/prep_dvm.ipynb```. The code for preprocessing UKBB will be uploaded soon, meanwhile we provide the code used to generate h5 files that we used for training given csv files with the tabular data and image paths. 




