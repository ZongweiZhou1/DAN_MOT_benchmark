Analysis on the method `DAN` for multi-target tracking.

We compare the `DAN` and `MOTDT` on the MOT16 and MOT17 brnchmark. It can be found that the `DAN` is more
dependent on the accuracy of the detection.

#### package dependence
- pytorch 0.3.1
- python 3.6
- motmetric 


# comparison
The experiments are based on the subsets :
```markdown
02, 05, 09, 11, 13

```
When we analyze on MOT16 benchmark, the results of `MOTDT` are:
```markdown
          IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP
MOT16-02 37.1% 75.6% 24.6% 30.2% 93.0%  54  7  21  26  406 12440  47  146 27.7% 0.247
MOT16-05 53.7% 83.0% 39.7% 44.6% 93.1% 125 13  68  44  224  3779  35  130 40.8% 0.242
MOT16-09 61.1% 75.8% 51.1% 63.6% 94.3%  25  8  15   2  202  1913  28   64 59.2% 0.247
MOT16-11 54.9% 72.2% 44.3% 58.1% 94.7%  69 12  28  29  301  3840  27   70 54.6% 0.208
MOT16-13 38.2% 71.6% 26.0% 29.7% 81.6% 107 11  38  58  766  8051  46  178 22.6% 0.276
OVERALL  46.1% 75.1% 33.3% 40.6% 91.5% 380 51 170 159 1899 30023 183  588 36.5% 0.241
```
and the results from `DAN` are:
```markdown
          IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs    FM  MOTA  MOTP                  │[I 11:15:37.862 NotebookApp] Saving file at /Code/
MOT16-02 22.4% 60.2% 13.7% 19.7% 86.3%  54  4  14  36  556 14319 165   298 15.7% 0.246                  │DAN/layer/sst.py
MOT16-05 41.2% 72.0% 28.9% 34.6% 86.4% 125  5  62  58  371  4457 122   203 27.4% 0.240                  │[I 11:28:05.235 NotebookApp] Saving file at /Code/
MOT16-09 47.9% 59.8% 40.0% 56.8% 84.9%  25  5  16   4  531  2270 139   201 44.1% 0.261                  │DAN/layer/sst_loss.py
MOT16-11 51.5% 66.9% 41.8% 53.9% 86.2%  69 10  27  32  792  4229  83   137 44.4% 0.214                  │[I 11:31:17.231 NotebookApp] Saving file at /Code/
MOT16-13 22.8% 71.3% 13.6% 16.9% 88.7% 107  6  28  73  246  9514 179   273 13.2% 0.264                  │DAN/layer/sst_loss.py
OVERALL  34.6% 65.3% 23.6% 31.2% 86.3% 380 30 147 203 2496 34789 688  1112 24.9% 0.240 

```
The `DAN` is much worser than 'MOTDT'.

However, when a better detector is selected, such as `SDP`, the conclusions become different.

These are the results of `MOTDT`
```markdown
              IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP
MOT17-02-DPM 36.8% 76.9% 24.2% 29.1% 92.6%  62  7  21  34  432 13172  45  147 26.5% 0.249
MOT17-05-DPM 51.6% 81.3% 37.8% 43.8% 94.0% 133 11  70  52  192  3887  34  135 40.5% 0.240
MOT17-09-DPM 59.0% 74.5% 48.8% 63.0% 96.2%  26  8  16   2  134  1972  35   69 59.8% 0.248
MOT17-11-DPM 54.1% 73.5% 42.8% 56.5% 96.9%  75 11  29  35  170  4107  30   80 54.4% 0.208
MOT17-13-DPM 37.8% 74.5% 25.3% 28.9% 85.1% 110 11  38  61  589  8275  46  183 23.5% 0.275
OVERALL      45.2% 75.9% 32.2% 39.5% 93.1% 406 48 174 184 1517 31413 190  614 36.2% 0.241

              IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML  FP    FN IDs   FM  MOTA  MOTP
MOT17-02-SDP 46.0% 73.6% 33.4% 43.4% 95.6%  62  8  34  20 373 10519 136  304 40.6% 0.190
MOT17-05-SDP 54.7% 78.3% 42.0% 51.2% 95.4% 133 21  78  34 172  3378  57  161 47.9% 0.175
MOT17-09-SDP 58.8% 74.1% 48.7% 64.5% 98.3%  26  9  15   2  61  1891  35   62 62.7% 0.141
MOT17-11-SDP 61.0% 79.6% 49.4% 61.2% 98.7%  75 10  38  27  77  3661  22  109 60.2% 0.135
MOT17-13-SDP 52.2% 81.7% 38.3% 44.5% 94.9% 110 26  37  47 280  6461  66  301 41.5% 0.211
OVERALL      52.8% 77.3% 40.1% 50.1% 96.4% 406 74 202 130 963 25910 316  937 47.6% 0.174

                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML  FP    FN IDs   FM  MOTA  MOTP
MOT17-02-FRCNN 41.0% 79.0% 27.7% 34.4% 98.3%  62  7  25  30 112 12187  54  159 33.5% 0.146
MOT17-05-FRCNN 51.6% 81.0% 37.9% 44.9% 95.9% 133 13  73  47 132  3813  25  111 42.6% 0.179
MOT17-09-FRCNN 58.8% 79.0% 46.9% 58.8% 99.1%  26  7  17   2  27  2192  18   48 58.0% 0.125
MOT17-11-FRCNN 56.4% 80.0% 43.6% 54.0% 98.9%  75  9  33  33  55  4344  14   74 53.2% 0.108
MOT17-13-FRCNN 54.5% 78.2% 41.9% 48.2% 90.1% 110 20  52  38 618  6029  84  332 42.2% 0.194
OVERALL        50.5% 79.3% 37.1% 45.0% 96.1% 406 56 200 150 944 28565 195  724 42.8% 0.150

```

and these are from 'DAN':
```markdown
              IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs    FM  MOTA  MOTP        
MOT17-02-DPM 21.6% 60.2% 13.2% 18.9% 86.3%  62  4  14  44  556 15067 165   298 15.0% 0.246        
MOT17-05-DPM 40.9% 72.2% 28.5% 34.2% 86.7% 133  5  62  66  364  4548 124   205 27.2% 0.240        
MOT17-09-DPM 47.6% 59.9% 39.5% 56.2% 85.3%  26  5  16   5  515  2330 137   200 44.0% 0.260        
MOT17-11-DPM 51.0% 68.1% 40.8% 52.5% 87.8%  75 10  27  38  687  4479  86   142 44.3% 0.215        
MOT17-13-DPM 22.5% 71.6% 13.4% 16.6% 89.2% 110  6  28  76  235  9706 179   273 13.1% 0.264        
OVERALL      34.1% 65.8% 23.0% 30.4% 87.0% 406 30 147 229 2357 36130 691  1118 24.5% 0.240


              IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN  IDs    FM  MOTA  MOTP       
MOT17-02-SDP 42.9% 62.7% 32.6% 47.5% 91.3%  62   7  36 19  837  9751  310   825 41.3% 0.197       
MOT17-05-SDP 64.0% 78.9% 53.8% 64.8% 94.9% 133  37  81 15  240  2438  203   256 58.3% 0.160       
MOT17-09-SDP 60.2% 75.9% 49.9% 65.0% 98.8%  26   8  17  1   41  1866   76   146 62.8% 0.145       
MOT17-11-SDP 67.0% 76.4% 59.6% 74.2% 95.2%  75  30  33 12  352  2431  117   235 69.3% 0.150       
MOT17-13-SDP 57.8% 72.9% 47.9% 57.4% 87.3% 110  47  24 39  972  4961  511   331 44.6% 0.213       
OVERALL      55.8% 71.9% 45.6% 58.7% 92.6% 406 129 191 86 2442 21447 1217  1793 51.6% 0.178


                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP            
MOT17-02-FRCNN 39.6% 72.7% 27.2% 35.1% 93.8%  62  7  26  29  432 12057 106  189 32.2% 0.123            
MOT17-05-FRCNN 60.7% 85.0% 47.2% 53.4% 96.2% 133 28  63  42  147  3223 110  101 49.7% 0.163            
MOT17-09-FRCNN 58.3% 80.0% 45.8% 56.8% 99.2%  26  7  17   2   24  2300  35   50 55.7% 0.090            
MOT17-11-FRCNN 56.1% 72.1% 45.9% 60.7% 95.3%  75 18  34  23  280  3710  64   76 57.0% 0.092            
MOT17-13-FRCNN 53.3% 63.6% 45.9% 58.8% 81.6% 110 36  52  22 1544  4798 545  443 40.8% 0.171            
OVERALL        51.0% 72.3% 39.4% 49.7% 91.4% 406 96 192 118 2427 26088 860  859 43.4% 0.130

```

Then the `DAN` is much better than `MOTDT`.


There are two main reasons:
- The MOTDT takes the prediction in account as candidates and a further confidence network is used
to filter out the detections with low confidence.
- The DAN only extracts the features from the center point of targets, and it also ignore the different
scales of different targets.