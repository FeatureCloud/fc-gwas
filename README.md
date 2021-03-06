# GWAS (sPLINK)

## Description
This app implements a federated GWAS as proposed in sPLINK. For now, only Chi-squared test is supported.

## Input
- `bed`
- `bim`
- `fam`
- `cov`


## Output
- `GWAS_results`: containing a table with all p-values
- `GWAS_result_plot`: containing the vulcano plot


## Workflows
This app is a standalone app and cannot be combined with any other FeatureCloud apps.

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```yml
fc_federated_GWAS:
  files:
    input: # names of the input BED/BIM/FAM/COV files
      bed: client1.bed
      bim: client1.bim
      fam: client1.fam
      cov: None # optional when choosing Chi-square; write nothing or 'None' when you do not want to use a COV-file.
    output:
      result: GWAS_result # name of the TXT file containing the numeric results
      result_plot: GWAS_result_plot # name of the PNG image showing the resulting plot (Manhattan plot)

  # parameters
  parameters:
    number_of_chunks: 0 # The number of parallel processes to be started. Choose '0' to let the program use the optimal number for your PC.

    # coordinator only:
    # algorithm: Chi-square # The algorithm to be used. Choose one of: 'Chi-square', 'Linear_regression' and 'Logarithmic_regression'.
    confounding_features: Sex,Age # The confounding features to be used, separated by a colon. The COV file has to contain these columns.

```

## Privacy
- No raw data is exchanged
- No privacy-enhancing techniques implemented yet