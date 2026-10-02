# ContinuousChunkingBehav

Code for [**Chunking improves working memory via compression, not direct retrieval from long-term memory**](https://osf.io/preprints/psyarxiv/46am8_v3)

## Authors

[Juan M. Chau](https://scholar.google.com/citations?hl=en&user=UA1kLj8AAAAJ)<sup>a</sup>, Joel Shebioba<sup>a</sup>, Anna Pitt<sup>a</sup>, [Matias J. Ison](https://scholar.google.co.uk/citations?user=2ULGtf8AAAAJ&hl=en)<sup>a</sup>, [Nicholas E. Myers](https://scholar.google.com/citations?user=4Ac4HK8AAAAJ&hl=en)<sup>a,b</sup><br><br>
<sup>a</sup> School of Psychology, University of Nottingham, Nottingham, UK<br>
<sup>b</sup> Department of Experimental Psychology & Oxford Centre for Human Brain Activity, University of Oxford, Oxford, UK<br>

## Instructions

Please download the *Experiment3_EEG* and *Experiment4_EEG* folders from the [study](https://osf.io/mh86y/) and place them at the same level as the *analysis* folder.

## Dependencies

MATLAB code was run on version R2025b, using [SPM12](https://github.com/spm/spm12)

Python code was run on version 3.9.13

| Package | Version |
|---|---|
| matplotlib | 3.9.4 |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| rpy2 | 3.5.17 |
| scipy | 1.13.1 |
| seaborn | 0.13.2 |
| scikit-learn | 1.6.1 |

R (v4.4.1) was accessed through rpy2, and was used for statistical analyses via the [jmv v2.5.6](https://cran.r-project.org/web/packages/jmv/index.html) package

## jlib

This repository includes a frozen snapshot of jlib in `analysis/`. It is the exact version used for the analyses in the paper and is not maintained here. Do not replace it with other future versions of jlib.
