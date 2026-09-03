# all_img_tests: per (condition, S) per task — mean±std (n=6 images, ddof=1)

## box_inpainting  (hole MSE | PSNR, mean±std over 6 images)
condition    S                          MSE       PSNR(dB)
baseline     spectral        0.1686±0.0758    19.99±2.27
xi_h0_full   spectral        0.1315±0.0821    21.62±3.67
3xi0_full    spectral        0.1149±0.0731    22.53±4.30
xi_h0_f20    spectral        0.1339±0.0762    21.42±3.47
3xi0_f20     spectral        0.1232±0.0727    22.18±4.28
baseline     pooled_junco    0.1752±0.0710    19.68±1.93
xi_h0_full   pooled_junco    0.1356±0.0723    21.12±2.91
3xi0_full    pooled_junco    0.1153±0.0689    22.40±4.19
xi_h0_f20    pooled_junco    0.1353±0.0717    21.12±2.90
3xi0_f20     pooled_junco    0.1134±0.0686    22.47±4.17

## random_inpainting  (hole MSE | PSNR, mean±std over 6 images)
condition    S                          MSE       PSNR(dB)
baseline     spectral        0.0213±0.0084    24.22±1.60
xi_h0_full   spectral        0.0161±0.0083    25.62±2.16
3xi0_full    spectral        0.0113±0.0077    27.74±3.30
xi_h0_f20    spectral        0.0161±0.0083    25.62±2.16
3xi0_f20     spectral        0.0113±0.0077    27.74±3.30
baseline     pooled_junco    0.0606±0.0178    19.75±1.16
xi_h0_full   pooled_junco    0.0314±0.0176    22.90±2.22
3xi0_full    pooled_junco    0.0189±0.0160    25.95±3.74
xi_h0_f20    pooled_junco    0.0314±0.0176    22.90±2.22
3xi0_f20     pooled_junco    0.0189±0.0160    25.95±3.74

## gaussian_blur  (full MSE | PSNR, mean±std over 6 images)
condition    S                          MSE       PSNR(dB)
baseline     spectral        0.0309±0.0117    21.36±1.53
xi_h0_full   spectral        0.0208±0.0114    23.35±2.27
3xi0_full    spectral        0.0147±0.0115    25.55±3.59
xi_h0_f20    spectral        0.0208±0.0114    23.35±2.27
3xi0_f20     spectral        0.0147±0.0115    25.55±3.59
baseline     pooled_junco    0.0652±0.0131    17.95±0.86
xi_h0_full   pooled_junco    0.0309±0.0118    21.37±1.55
3xi0_full    pooled_junco    0.0148±0.0116    25.52±3.61
xi_h0_f20    pooled_junco    0.0309±0.0118    21.37±1.55
3xi0_f20     pooled_junco    0.0148±0.0116    25.52±3.61

## motion_blur  (full MSE | PSNR, mean±std over 6 images)
condition    S                          MSE       PSNR(dB)
baseline     spectral        0.0330±0.0103    20.99±1.27
xi_h0_full   spectral        0.0214±0.0097    23.08±1.90
3xi0_full    spectral        0.0145±0.0095    25.25±2.97
xi_h0_f20    spectral        0.0214±0.0097    23.08±1.90
3xi0_f20     spectral        0.0145±0.0095    25.25±2.97
baseline     pooled_junco    0.0644±0.0113    17.99±0.76
xi_h0_full   pooled_junco    0.0319±0.0100    21.15±1.29
3xi0_full    pooled_junco    0.0148±0.0095    25.07±2.82
xi_h0_f20    pooled_junco    0.0319±0.0100    21.15±1.29
3xi0_f20     pooled_junco    0.0148±0.0094    25.07±2.81

## superresolution  (full MSE | PSNR, mean±std over 6 images)
condition    S                          MSE       PSNR(dB)
baseline     spectral        0.0294±0.0100    21.53±1.39
xi_h0_full   spectral        0.0203±0.0099    23.37±2.04
3xi0_full    spectral        0.0141±0.0101    25.50±3.23
xi_h0_f20    spectral        0.0203±0.0099    23.37±2.04
3xi0_f20     spectral        0.0141±0.0101    25.50±3.23
baseline     pooled_junco    0.0646±0.0108    17.97±0.73
xi_h0_full   pooled_junco    0.0297±0.0096    21.46±1.34
3xi0_full    pooled_junco    0.0137±0.0096    25.63±3.24
xi_h0_f20    pooled_junco    0.0297±0.0096    21.46±1.34
3xi0_f20     pooled_junco    0.0137±0.0096    25.63±3.24
