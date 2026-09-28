# ISIC2018 Task 3 Dataset Statistics

## Summary
```text
     split  total_samples  matched_images  unique_lesions
      test           1512            1512            1223
     train          10015           10015            7470
validation            193             193             145
```

## Binary Counts
```text
     split  target binary_label  count
      test       0       benign   1224
      test       1    malignant    288
     train       0       benign   8191
     train       1    malignant   1824
validation       0       benign    149
validation       1    malignant     44
```

## Diagnosis Counts
```text
     split  class_index                  diagnosis  count
      test            0          actinic keratosis     19
      test            1       basal cell carcinoma     93
      test            2             dermatofibroma     44
      test            3                   melanoma    171
      test            4                      nevus    909
      test            5 pigmented benign keratosis    217
      test            6    squamous cell carcinoma     24
      test            7            vascular lesion     35
     train            0          actinic keratosis    130
     train            1       basal cell carcinoma    514
     train            2             dermatofibroma    115
     train            3                   melanoma   1113
     train            4                      nevus   6705
     train            5 pigmented benign keratosis   1099
     train            6    squamous cell carcinoma    197
     train            7            vascular lesion    142
validation            1       basal cell carcinoma     15
validation            2             dermatofibroma      1
validation            3                   melanoma     21
validation            4                      nevus    123
validation            5 pigmented benign keratosis     22
validation            6    squamous cell carcinoma      8
validation            7            vascular lesion      3
```

## Missing Values
```text
     split               field  missing_count
      test          age_approx            288
      test                 sex            295
      test anatom_site_general            491
      test    benign_malignant            432
      test           diagnosis              0
      test              target              0
      test           lesion_id              0
      test          image_file              0
     train          age_approx             94
     train                 sex             47
     train anatom_site_general           1686
     train    benign_malignant           2197
     train           diagnosis              0
     train              target              0
     train           lesion_id              0
     train          image_file              0
validation          age_approx              1
validation                 sex              1
validation anatom_site_general             40
validation    benign_malignant             49
validation           diagnosis              0
validation              target              0
validation           lesion_id              0
validation          image_file              0
```

## Label Mapping
```text
 class_index                  diagnosis
           0          actinic keratosis
           1       basal cell carcinoma
           2             dermatofibroma
           3                   melanoma
           4                      nevus
           5 pigmented benign keratosis
           6    squamous cell carcinoma
           7            vascular lesion
```