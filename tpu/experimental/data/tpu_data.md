================================================================================================
WHEN TO SPLIT / BEST LEAF   b=128   backend=TPU   (time = median of 25 blocked runs)
  For each N we sweep LEAF = k*b (k=1..32) + direct.  Splitting stops when a
  subproblem <= LEAF.  work = MXU-padded flops (exact).  LOWER work/time = better.
  CPU time is indicative only; the 128x128 cliff is real only on TPU. Trust work.
================================================================================================

################################################################################################
#  0 splits  (N <= b, direct only)
################################################################################################

  N = 8   (max splits @ LEAF=b: 0)
         LEAF   =k*b levels  leaf_sz      work         time   best
            8 direct      1        8     16.4K     206.1 us   min-WORK min-TIME

  N = 16   (max splits @ LEAF=b: 0)
         LEAF   =k*b levels  leaf_sz      work         time   best
           16 direct      1       16     16.4K     188.3 us   min-WORK min-TIME

  N = 32   (max splits @ LEAF=b: 0)
         LEAF   =k*b levels  leaf_sz      work         time   best
           32 direct      1       32     16.4K     218.1 us   min-WORK min-TIME

  N = 48   (max splits @ LEAF=b: 0)
         LEAF   =k*b levels  leaf_sz      work         time   best
           48 direct      1       48     16.4K     175.5 us   min-WORK min-TIME

  N = 64   (max splits @ LEAF=b: 0)
         LEAF   =k*b levels  leaf_sz      work         time   best
           64 direct      1       64     16.4K     152.2 us   min-WORK min-TIME

  N = 96   (max splits @ LEAF=b: 0)
         LEAF   =k*b levels  leaf_sz      work         time   best
           96 direct      1       96     16.4K     211.0 us   min-WORK min-TIME

  N = 128   (max splits @ LEAF=b: 0)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128 direct      1      128     16.4K     226.6 us   min-WORK min-TIME

################################################################################################
#  1 split   (N = b   * k)
################################################################################################

  N = 256   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2        2      2.1M     208.0 us   min-TIME
          256 direct      1      256     65.5K     282.9 us   min-WORK

  N = 384   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2        3      2.1M     216.4 us   min-TIME
          256     2b      2        3      2.1M     216.4 us   
          384 direct      1      384    147.5K     389.5 us   min-WORK

  N = 512   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2        4      2.2M     178.9 us   min-TIME
          256     2b      2        4      2.2M     178.9 us   
          384     3b      2        4      2.2M     178.9 us   
          512 direct      1      512    262.1K     499.5 us   min-WORK

  N = 640   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2        5      2.2M     179.1 us   min-TIME
          256     2b      2        5      2.2M     179.1 us   
          384     3b      2        5      2.2M     179.1 us   
          512     4b      2        5      2.2M     179.1 us   
          640 direct      1      640    409.6K     649.5 us   min-WORK

  N = 768   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2        6      2.2M     167.5 us   min-TIME
          256     2b      2        6      2.2M     167.5 us   
          384     3b      2        6      2.2M     167.5 us   
          512     4b      2        6      2.2M     167.5 us   
          640     5b      2        6      2.2M     167.5 us   
          768 direct      1      768    589.8K     861.6 us   min-WORK

  N = 1,024   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2        8      2.2M     213.2 us   min-TIME
          256     2b      2        8      2.2M     213.2 us   
          384     3b      2        8      2.2M     213.2 us   
          512     4b      2        8      2.2M     213.2 us   
          640     5b      2        8      2.2M     213.2 us   
          768     6b      2        8      2.2M     213.2 us   
          896     7b      2        8      2.2M     213.2 us   
        1,024 direct      1     1024      1.0M      1.67 ms   min-WORK

  N = 1,280   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       10      2.3M     200.7 us   min-TIME
          256     2b      2       10      2.3M     200.7 us   
          384     3b      2       10      2.3M     200.7 us   
          512     4b      2       10      2.3M     200.7 us   
          640     5b      2       10      2.3M     200.7 us   
          768     6b      2       10      2.3M     200.7 us   
          896     7b      2       10      2.3M     200.7 us   
        1,024     8b      2       10      2.3M     200.7 us   
        1,152     9b      2       10      2.3M     200.7 us   
        1,280 direct      1     1280      1.6M      2.74 ms   min-WORK

  N = 1,408   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       11      2.3M     216.4 us   min-TIME
          256     2b      2       11      2.3M     216.4 us   
          384     3b      2       11      2.3M     216.4 us   
          512     4b      2       11      2.3M     216.4 us   
          640     5b      2       11      2.3M     216.4 us   
          768     6b      2       11      2.3M     216.4 us   
          896     7b      2       11      2.3M     216.4 us   
        1,024     8b      2       11      2.3M     216.4 us   
        1,152     9b      2       11      2.3M     216.4 us   
        1,280    10b      2       11      2.3M     216.4 us   
        1,408 direct      1     1408      2.0M      3.51 ms   min-WORK

  N = 1,536   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       12      2.3M     223.3 us   min-WORK min-TIME
          256     2b      2       12      2.3M     223.3 us   
          384     3b      2       12      2.3M     223.3 us   
          512     4b      2       12      2.3M     223.3 us   
          640     5b      2       12      2.3M     223.3 us   
          768     6b      2       12      2.3M     223.3 us   
          896     7b      2       12      2.3M     223.3 us   
        1,024     8b      2       12      2.3M     223.3 us   
        1,152     9b      2       12      2.3M     223.3 us   
        1,280    10b      2       12      2.3M     223.3 us   
        1,408    11b      2       12      2.3M     223.3 us   
        1,536 direct      1     1536      2.4M      4.11 ms   

  N = 1,664   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       13      2.3M     204.9 us   min-WORK min-TIME
          256     2b      2       13      2.3M     204.9 us   
          384     3b      2       13      2.3M     204.9 us   
          512     4b      2       13      2.3M     204.9 us   
          640     5b      2       13      2.3M     204.9 us   
          768     6b      2       13      2.3M     204.9 us   
          896     7b      2       13      2.3M     204.9 us   
        1,024     8b      2       13      2.3M     204.9 us   
        1,152     9b      2       13      2.3M     204.9 us   
        1,280    10b      2       13      2.3M     204.9 us   
        1,408    11b      2       13      2.3M     204.9 us   
        1,536    12b      2       13      2.3M     204.9 us   
        1,664 direct      1     1664      2.8M      4.78 ms   

  N = 1,792   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       14      2.3M     227.5 us   min-WORK min-TIME
          256     2b      2       14      2.3M     227.5 us   
          384     3b      2       14      2.3M     227.5 us   
          512     4b      2       14      2.3M     227.5 us   
          640     5b      2       14      2.3M     227.5 us   
          768     6b      2       14      2.3M     227.5 us   
          896     7b      2       14      2.3M     227.5 us   
        1,024     8b      2       14      2.3M     227.5 us   
        1,152     9b      2       14      2.3M     227.5 us   
        1,280    10b      2       14      2.3M     227.5 us   
        1,408    11b      2       14      2.3M     227.5 us   
        1,536    12b      2       14      2.3M     227.5 us   
        1,664    13b      2       14      2.3M     227.5 us   
        1,792 direct      1     1792      3.2M      5.95 ms   

  N = 1,920   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       15      2.3M     225.3 us   min-WORK min-TIME
          256     2b      2       15      2.3M     225.3 us   
          384     3b      2       15      2.3M     225.3 us   
          512     4b      2       15      2.3M     225.3 us   
          640     5b      2       15      2.3M     225.3 us   
          768     6b      2       15      2.3M     225.3 us   
          896     7b      2       15      2.3M     225.3 us   
        1,024     8b      2       15      2.3M     225.3 us   
        1,152     9b      2       15      2.3M     225.3 us   
        1,280    10b      2       15      2.3M     225.3 us   
        1,408    11b      2       15      2.3M     225.3 us   
        1,536    12b      2       15      2.3M     225.3 us   
        1,664    13b      2       15      2.3M     225.3 us   
        1,792    14b      2       15      2.3M     225.3 us   
        1,920 direct      1     1920      3.7M      6.91 ms   

  N = 2,048   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       16      2.4M     250.8 us   min-WORK min-TIME
          256     2b      2       16      2.4M     250.8 us   
          384     3b      2       16      2.4M     250.8 us   
          512     4b      2       16      2.4M     250.8 us   
          640     5b      2       16      2.4M     250.8 us   
          768     6b      2       16      2.4M     250.8 us   
          896     7b      2       16      2.4M     250.8 us   
        1,024     8b      2       16      2.4M     250.8 us   
        1,152     9b      2       16      2.4M     250.8 us   
        1,280    10b      2       16      2.4M     250.8 us   
        1,408    11b      2       16      2.4M     250.8 us   
        1,536    12b      2       16      2.4M     250.8 us   
        1,664    13b      2       16      2.4M     250.8 us   
        1,792    14b      2       16      2.4M     250.8 us   
        1,920    15b      2       16      2.4M     250.8 us   
        2,048 direct      1     2048      4.2M      8.74 ms   

  N = 2,560   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       20      2.4M     193.6 us   min-WORK min-TIME
          256     2b      2       20      2.4M     193.6 us   
          384     3b      2       20      2.4M     193.6 us   
          512     4b      2       20      2.4M     193.6 us   
          640     5b      2       20      2.4M     193.6 us   
          768     6b      2       20      2.4M     193.6 us   
          896     7b      2       20      2.4M     193.6 us   
        1,024     8b      2       20      2.4M     193.6 us   
        1,152     9b      2       20      2.4M     193.6 us   
        1,280    10b      2       20      2.4M     193.6 us   
        1,408    11b      2       20      2.4M     193.6 us   
        1,536    12b      2       20      2.4M     193.6 us   
        1,664    13b      2       20      2.4M     193.6 us   
        1,792    14b      2       20      2.4M     193.6 us   
        1,920    15b      2       20      2.4M     193.6 us   
        2,048    16b      2       20      2.4M     193.6 us   
        2,176    17b      2       20      2.4M     193.6 us   
        2,304    18b      2       20      2.4M     193.6 us   
        2,432    19b      2       20      2.4M     193.6 us   
        2,560 direct      1     2560      6.6M     12.86 ms   

  N = 3,072   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       24      2.5M     230.5 us   min-WORK min-TIME
          256     2b      2       24      2.5M     230.5 us   
          384     3b      2       24      2.5M     230.5 us   
          512     4b      2       24      2.5M     230.5 us   
          640     5b      2       24      2.5M     230.5 us   
          768     6b      2       24      2.5M     230.5 us   
          896     7b      2       24      2.5M     230.5 us   
        1,024     8b      2       24      2.5M     230.5 us   
        1,152     9b      2       24      2.5M     230.5 us   
        1,280    10b      2       24      2.5M     230.5 us   
        1,408    11b      2       24      2.5M     230.5 us   
        1,536    12b      2       24      2.5M     230.5 us   
        1,664    13b      2       24      2.5M     230.5 us   
        1,792    14b      2       24      2.5M     230.5 us   
        1,920    15b      2       24      2.5M     230.5 us   
        2,048    16b      2       24      2.5M     230.5 us   
        2,176    17b      2       24      2.5M     230.5 us   
        2,304    18b      2       24      2.5M     230.5 us   
        2,432    19b      2       24      2.5M     230.5 us   
        2,560    20b      2       24      2.5M     230.5 us   
        2,688    21b      2       24      2.5M     230.5 us   
        2,816    22b      2       24      2.5M     230.5 us   
        2,944    23b      2       24      2.5M     230.5 us   
        3,072 direct      1     3072      9.4M     18.32 ms   

  N = 3,200   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       25      2.5M     235.1 us   min-WORK min-TIME
          256     2b      2       25      2.5M     235.1 us   
          384     3b      2       25      2.5M     235.1 us   
          512     4b      2       25      2.5M     235.1 us   
          640     5b      2       25      2.5M     235.1 us   
          768     6b      2       25      2.5M     235.1 us   
          896     7b      2       25      2.5M     235.1 us   
        1,024     8b      2       25      2.5M     235.1 us   
        1,152     9b      2       25      2.5M     235.1 us   
        1,280    10b      2       25      2.5M     235.1 us   
        1,408    11b      2       25      2.5M     235.1 us   
        1,536    12b      2       25      2.5M     235.1 us   
        1,664    13b      2       25      2.5M     235.1 us   
        1,792    14b      2       25      2.5M     235.1 us   
        1,920    15b      2       25      2.5M     235.1 us   
        2,048    16b      2       25      2.5M     235.1 us   
        2,176    17b      2       25      2.5M     235.1 us   
        2,304    18b      2       25      2.5M     235.1 us   
        2,432    19b      2       25      2.5M     235.1 us   
        2,560    20b      2       25      2.5M     235.1 us   
        2,688    21b      2       25      2.5M     235.1 us   
        2,816    22b      2       25      2.5M     235.1 us   
        2,944    23b      2       25      2.5M     235.1 us   
        3,072    24b      2       25      2.5M     235.1 us   
        3,200 direct      1     3200     10.2M     19.74 ms   

  N = 3,328   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       26      2.5M     164.5 us   min-WORK min-TIME
          256     2b      2       26      2.5M     164.5 us   
          384     3b      2       26      2.5M     164.5 us   
          512     4b      2       26      2.5M     164.5 us   
          640     5b      2       26      2.5M     164.5 us   
          768     6b      2       26      2.5M     164.5 us   
          896     7b      2       26      2.5M     164.5 us   
        1,024     8b      2       26      2.5M     164.5 us   
        1,152     9b      2       26      2.5M     164.5 us   
        1,280    10b      2       26      2.5M     164.5 us   
        1,408    11b      2       26      2.5M     164.5 us   
        1,536    12b      2       26      2.5M     164.5 us   
        1,664    13b      2       26      2.5M     164.5 us   
        1,792    14b      2       26      2.5M     164.5 us   
        1,920    15b      2       26      2.5M     164.5 us   
        2,048    16b      2       26      2.5M     164.5 us   
        2,176    17b      2       26      2.5M     164.5 us   
        2,304    18b      2       26      2.5M     164.5 us   
        2,432    19b      2       26      2.5M     164.5 us   
        2,560    20b      2       26      2.5M     164.5 us   
        2,688    21b      2       26      2.5M     164.5 us   
        2,816    22b      2       26      2.5M     164.5 us   
        2,944    23b      2       26      2.5M     164.5 us   
        3,072    24b      2       26      2.5M     164.5 us   
        3,200    25b      2       26      2.5M     164.5 us   
        3,328 direct      1     3328     11.1M     21.60 ms   

  N = 3,456   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       27      2.5M     181.1 us   min-WORK min-TIME
          256     2b      2       27      2.5M     181.1 us   
          384     3b      2       27      2.5M     181.1 us   
          512     4b      2       27      2.5M     181.1 us   
          640     5b      2       27      2.5M     181.1 us   
          768     6b      2       27      2.5M     181.1 us   
          896     7b      2       27      2.5M     181.1 us   
        1,024     8b      2       27      2.5M     181.1 us   
        1,152     9b      2       27      2.5M     181.1 us   
        1,280    10b      2       27      2.5M     181.1 us   
        1,408    11b      2       27      2.5M     181.1 us   
        1,536    12b      2       27      2.5M     181.1 us   
        1,664    13b      2       27      2.5M     181.1 us   
        1,792    14b      2       27      2.5M     181.1 us   
        1,920    15b      2       27      2.5M     181.1 us   
        2,048    16b      2       27      2.5M     181.1 us   
        2,176    17b      2       27      2.5M     181.1 us   
        2,304    18b      2       27      2.5M     181.1 us   
        2,432    19b      2       27      2.5M     181.1 us   
        2,560    20b      2       27      2.5M     181.1 us   
        2,688    21b      2       27      2.5M     181.1 us   
        2,816    22b      2       27      2.5M     181.1 us   
        2,944    23b      2       27      2.5M     181.1 us   
        3,072    24b      2       27      2.5M     181.1 us   
        3,200    25b      2       27      2.5M     181.1 us   
        3,328    26b      2       27      2.5M     181.1 us   
        3,456 direct      1     3456     11.9M     23.45 ms   

 N = 3,584   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       28      2.6M     174.9 us   min-WORK min-TIME
          256     2b      2       28      2.6M     174.9 us   
          384     3b      2       28      2.6M     174.9 us   
          512     4b      2       28      2.6M     174.9 us   
          640     5b      2       28      2.6M     174.9 us   
          768     6b      2       28      2.6M     174.9 us   
          896     7b      2       28      2.6M     174.9 us   
        1,024     8b      2       28      2.6M     174.9 us   
        1,152     9b      2       28      2.6M     174.9 us   
        1,280    10b      2       28      2.6M     174.9 us   
        1,408    11b      2       28      2.6M     174.9 us   
        1,536    12b      2       28      2.6M     174.9 us   
        1,664    13b      2       28      2.6M     174.9 us   
        1,792    14b      2       28      2.6M     174.9 us   
        1,920    15b      2       28      2.6M     174.9 us   
        2,048    16b      2       28      2.6M     174.9 us   
        2,176    17b      2       28      2.6M     174.9 us   
        2,304    18b      2       28      2.6M     174.9 us   
        2,432    19b      2       28      2.6M     174.9 us   
        2,560    20b      2       28      2.6M     174.9 us   
        2,688    21b      2       28      2.6M     174.9 us   
        2,816    22b      2       28      2.6M     174.9 us   
        2,944    23b      2       28      2.6M     174.9 us   
        3,072    24b      2       28      2.6M     174.9 us   
        3,200    25b      2       28      2.6M     174.9 us   
        3,328    26b      2       28      2.6M     174.9 us   
        3,456    27b      2       28      2.6M     174.9 us   
        3,584 direct      1     3584     12.8M     24.97 ms   

  N = 4,096   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       32      2.6M     196.4 us   min-WORK min-TIME
          256     2b      2       32      2.6M     196.4 us   
          384     3b      2       32      2.6M     196.4 us   
          512     4b      2       32      2.6M     196.4 us   
          640     5b      2       32      2.6M     196.4 us   
          768     6b      2       32      2.6M     196.4 us   
          896     7b      2       32      2.6M     196.4 us   
        1,024     8b      2       32      2.6M     196.4 us   
        1,152     9b      2       32      2.6M     196.4 us   
        1,280    10b      2       32      2.6M     196.4 us   
        1,408    11b      2       32      2.6M     196.4 us   
        1,536    12b      2       32      2.6M     196.4 us   
        1,664    13b      2       32      2.6M     196.4 us   
        1,792    14b      2       32      2.6M     196.4 us   
        1,920    15b      2       32      2.6M     196.4 us   
        2,048    16b      2       32      2.6M     196.4 us   
        2,176    17b      2       32      2.6M     196.4 us   
        2,304    18b      2       32      2.6M     196.4 us   
        2,432    19b      2       32      2.6M     196.4 us   
        2,560    20b      2       32      2.6M     196.4 us   
        2,688    21b      2       32      2.6M     196.4 us   
        2,816    22b      2       32      2.6M     196.4 us   
        2,944    23b      2       32      2.6M     196.4 us   
        3,072    24b      2       32      2.6M     196.4 us   
        3,200    25b      2       32      2.6M     196.4 us   
        3,328    26b      2       32      2.6M     196.4 us   
        3,456    27b      2       32      2.6M     196.4 us   
        3,584    28b      2       32      2.6M     196.4 us   
        3,712    29b      2       32      2.6M     196.4 us   
        3,840    30b      2       32      2.6M     196.4 us   
        3,968    31b      2       32      2.6M     196.4 us   
        4,096 direct      1     4096     16.8M     33.80 ms   

  N = 5,120   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       40      2.8M     211.6 us   min-WORK min-TIME
          256     2b      2       40      2.8M     211.6 us   
          384     3b      2       40      2.8M     211.6 us   
          512     4b      2       40      2.8M     211.6 us   
          640     5b      2       40      2.8M     211.6 us   
          768     6b      2       40      2.8M     211.6 us   
          896     7b      2       40      2.8M     211.6 us   
        1,024     8b      2       40      2.8M     211.6 us   
        1,152     9b      2       40      2.8M     211.6 us   
        1,280    10b      2       40      2.8M     211.6 us   
        1,408    11b      2       40      2.8M     211.6 us   
        1,536    12b      2       40      2.8M     211.6 us   
        1,664    13b      2       40      2.8M     211.6 us   
        1,792    14b      2       40      2.8M     211.6 us   
        1,920    15b      2       40      2.8M     211.6 us   
        2,048    16b      2       40      2.8M     211.6 us   
        2,176    17b      2       40      2.8M     211.6 us   
        2,304    18b      2       40      2.8M     211.6 us   
        2,432    19b      2       40      2.8M     211.6 us   
        2,560    20b      2       40      2.8M     211.6 us   
        2,688    21b      2       40      2.8M     211.6 us   
        2,816    22b      2       40      2.8M     211.6 us   
        2,944    23b      2       40      2.8M     211.6 us   
        3,072    24b      2       40      2.8M     211.6 us   
        3,200    25b      2       40      2.8M     211.6 us   
        3,328    26b      2       40      2.8M     211.6 us   
        3,456    27b      2       40      2.8M     211.6 us   
        3,584    28b      2       40      2.8M     211.6 us   
        3,712    29b      2       40      2.8M     211.6 us   
        3,840    30b      2       40      2.8M     211.6 us   
        3,968    31b      2       40      2.8M     211.6 us   
        4,096    32b      2       40      2.8M     211.6 us   
        5,120 direct      1     5120     26.2M    (too big)   

  N = 6,144   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       48      2.9M     216.2 us   min-WORK min-TIME
          256     2b      2       48      2.9M     216.2 us   
          384     3b      2       48      2.9M     216.2 us   
          512     4b      2       48      2.9M     216.2 us   
          640     5b      2       48      2.9M     216.2 us   
          768     6b      2       48      2.9M     216.2 us   
          896     7b      2       48      2.9M     216.2 us   
        1,024     8b      2       48      2.9M     216.2 us   
        1,152     9b      2       48      2.9M     216.2 us   
        1,280    10b      2       48      2.9M     216.2 us   
        1,408    11b      2       48      2.9M     216.2 us   
        1,536    12b      2       48      2.9M     216.2 us   
        1,664    13b      2       48      2.9M     216.2 us   
        1,792    14b      2       48      2.9M     216.2 us   
        1,920    15b      2       48      2.9M     216.2 us   
        2,048    16b      2       48      2.9M     216.2 us   
        2,176    17b      2       48      2.9M     216.2 us   
        2,304    18b      2       48      2.9M     216.2 us   
        2,432    19b      2       48      2.9M     216.2 us   
        2,560    20b      2       48      2.9M     216.2 us   
        2,688    21b      2       48      2.9M     216.2 us   
        2,816    22b      2       48      2.9M     216.2 us   
        2,944    23b      2       48      2.9M     216.2 us   
        3,072    24b      2       48      2.9M     216.2 us   
        3,200    25b      2       48      2.9M     216.2 us   
        3,328    26b      2       48      2.9M     216.2 us   
        3,456    27b      2       48      2.9M     216.2 us   
        3,584    28b      2       48      2.9M     216.2 us   
        3,712    29b      2       48      2.9M     216.2 us   
        3,840    30b      2       48      2.9M     216.2 us   
        3,968    31b      2       48      2.9M     216.2 us   
        4,096    32b      2       48      2.9M     216.2 us   
        6,144 direct      1     6144     37.7M    (too big)   

  N = 8,192   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       64      3.1M     225.4 us   min-WORK min-TIME
          256     2b      2       64      3.1M     225.4 us   
          384     3b      2       64      3.1M     225.4 us   
          512     4b      2       64      3.1M     225.4 us   
          640     5b      2       64      3.1M     225.4 us   
          768     6b      2       64      3.1M     225.4 us   
          896     7b      2       64      3.1M     225.4 us   
        1,024     8b      2       64      3.1M     225.4 us   
        1,152     9b      2       64      3.1M     225.4 us   
        1,280    10b      2       64      3.1M     225.4 us   
        1,408    11b      2       64      3.1M     225.4 us   
        1,536    12b      2       64      3.1M     225.4 us   
        1,664    13b      2       64      3.1M     225.4 us   
        1,792    14b      2       64      3.1M     225.4 us   
        1,920    15b      2       64      3.1M     225.4 us   
        2,048    16b      2       64      3.1M     225.4 us   
        2,176    17b      2       64      3.1M     225.4 us   
        2,304    18b      2       64      3.1M     225.4 us   
        2,432    19b      2       64      3.1M     225.4 us   
        2,560    20b      2       64      3.1M     225.4 us   
        2,688    21b      2       64      3.1M     225.4 us   
        2,816    22b      2       64      3.1M     225.4 us   
        2,944    23b      2       64      3.1M     225.4 us   
        3,072    24b      2       64      3.1M     225.4 us   
        3,200    25b      2       64      3.1M     225.4 us   
        3,328    26b      2       64      3.1M     225.4 us   
        3,456    27b      2       64      3.1M     225.4 us   
        3,584    28b      2       64      3.1M     225.4 us   
        3,712    29b      2       64      3.1M     225.4 us   
        3,840    30b      2       64      3.1M     225.4 us   
        3,968    31b      2       64      3.1M     225.4 us   
        4,096    32b      2       64      3.1M     225.4 us   
        8,192 direct      1     8192     67.1M    (too big)   

  N = 12,288   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2       96      3.7M     255.7 us   min-WORK min-TIME
          256     2b      2       96      3.7M     255.7 us   
          384     3b      2       96      3.7M     255.7 us   
          512     4b      2       96      3.7M     255.7 us   
          640     5b      2       96      3.7M     255.7 us   
          768     6b      2       96      3.7M     255.7 us   
          896     7b      2       96      3.7M     255.7 us   
        1,024     8b      2       96      3.7M     255.7 us   
        1,152     9b      2       96      3.7M     255.7 us   
        1,280    10b      2       96      3.7M     255.7 us   
        1,408    11b      2       96      3.7M     255.7 us   
        1,536    12b      2       96      3.7M     255.7 us   
        1,664    13b      2       96      3.7M     255.7 us   
        1,792    14b      2       96      3.7M     255.7 us   
        1,920    15b      2       96      3.7M     255.7 us   
        2,048    16b      2       96      3.7M     255.7 us   
        2,176    17b      2       96      3.7M     255.7 us   
        2,304    18b      2       96      3.7M     255.7 us   
        2,432    19b      2       96      3.7M     255.7 us   
        2,560    20b      2       96      3.7M     255.7 us   
        2,688    21b      2       96      3.7M     255.7 us   
        2,816    22b      2       96      3.7M     255.7 us   
        2,944    23b      2       96      3.7M     255.7 us   
        3,072    24b      2       96      3.7M     255.7 us   
        3,200    25b      2       96      3.7M     255.7 us   
        3,328    26b      2       96      3.7M     255.7 us   
        3,456    27b      2       96      3.7M     255.7 us   
        3,584    28b      2       96      3.7M     255.7 us   
        3,712    29b      2       96      3.7M     255.7 us   
        3,840    30b      2       96      3.7M     255.7 us   
        3,968    31b      2       96      3.7M     255.7 us   
        4,096    32b      2       96      3.7M     255.7 us   
       12,288 direct      1    12288    151.0M    (too big)   

  N = 16,384   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2      128      4.2M     225.6 us   min-WORK min-TIME
          256     2b      2      128      4.2M     225.6 us   
          384     3b      2      128      4.2M     225.6 us   
          512     4b      2      128      4.2M     225.6 us   
          640     5b      2      128      4.2M     225.6 us   
          768     6b      2      128      4.2M     225.6 us   
          896     7b      2      128      4.2M     225.6 us   
        1,024     8b      2      128      4.2M     225.6 us   
        1,152     9b      2      128      4.2M     225.6 us   
        1,280    10b      2      128      4.2M     225.6 us   
        1,408    11b      2      128      4.2M     225.6 us   
        1,536    12b      2      128      4.2M     225.6 us   
        1,664    13b      2      128      4.2M     225.6 us   
        1,792    14b      2      128      4.2M     225.6 us   
        1,920    15b      2      128      4.2M     225.6 us   
        2,048    16b      2      128      4.2M     225.6 us   
        2,176    17b      2      128      4.2M     225.6 us   
        2,304    18b      2      128      4.2M     225.6 us   
        2,432    19b      2      128      4.2M     225.6 us   
        2,560    20b      2      128      4.2M     225.6 us   
        2,688    21b      2      128      4.2M     225.6 us   
        2,816    22b      2      128      4.2M     225.6 us   
        2,944    23b      2      128      4.2M     225.6 us   
        3,072    24b      2      128      4.2M     225.6 us   
        3,200    25b      2      128      4.2M     225.6 us   
        3,328    26b      2      128      4.2M     225.6 us   
        3,456    27b      2      128      4.2M     225.6 us   
        3,584    28b      2      128      4.2M     225.6 us   
        3,712    29b      2      128      4.2M     225.6 us   
        3,840    30b      2      128      4.2M     225.6 us   
        3,968    31b      2      128      4.2M     225.6 us   
        4,096    32b      2      128      4.2M     225.6 us   
       16,384 direct      1    16384    268.4M    (too big)   

################################################################################################
#  2 splits  (N = b^2 * k)
################################################################################################

  N = 16,384   (max splits @ LEAF=b: 1)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      2      128      4.2M     232.1 us   min-WORK min-TIME
          256     2b      2      128      4.2M     232.1 us   
          384     3b      2      128      4.2M     232.1 us   
          512     4b      2      128      4.2M     232.1 us   
          640     5b      2      128      4.2M     232.1 us   
          768     6b      2      128      4.2M     232.1 us   
          896     7b      2      128      4.2M     232.1 us   
        1,024     8b      2      128      4.2M     232.1 us   
        1,152     9b      2      128      4.2M     232.1 us   
        1,280    10b      2      128      4.2M     232.1 us   
        1,408    11b      2      128      4.2M     232.1 us   
        1,536    12b      2      128      4.2M     232.1 us   
        1,664    13b      2      128      4.2M     232.1 us   
        1,792    14b      2      128      4.2M     232.1 us   
        1,920    15b      2      128      4.2M     232.1 us   
        2,048    16b      2      128      4.2M     232.1 us   
        2,176    17b      2      128      4.2M     232.1 us   
        2,304    18b      2      128      4.2M     232.1 us   
        2,432    19b      2      128      4.2M     232.1 us   
        2,560    20b      2      128      4.2M     232.1 us   
        2,688    21b      2      128      4.2M     232.1 us   
        2,816    22b      2      128      4.2M     232.1 us   
        2,944    23b      2      128      4.2M     232.1 us   
        3,072    24b      2      128      4.2M     232.1 us   
        3,200    25b      2      128      4.2M     232.1 us   
        3,328    26b      2      128      4.2M     232.1 us   
        3,456    27b      2      128      4.2M     232.1 us   
        3,584    28b      2      128      4.2M     232.1 us   
        3,712    29b      2      128      4.2M     232.1 us   
        3,840    30b      2      128      4.2M     232.1 us   
        3,968    31b      2      128      4.2M     232.1 us   
        4,096    32b      2      128      4.2M     232.1 us   
       16,384 direct      1    16384    268.4M    (too big)   

  N = 32,768   (max splits @ LEAF=b: 2)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      3        2    276.8M     244.3 us   min-TIME
          256     2b      2      256     12.6M     313.8 us   min-WORK
          384     3b      2      256     12.6M     313.8 us   
          512     4b      2      256     12.6M     313.8 us   
          640     5b      2      256     12.6M     313.8 us   
          768     6b      2      256     12.6M     313.8 us   
          896     7b      2      256     12.6M     313.8 us   
        1,024     8b      2      256     12.6M     313.8 us   
        1,152     9b      2      256     12.6M     313.8 us   
        1,280    10b      2      256     12.6M     313.8 us   
        1,408    11b      2      256     12.6M     313.8 us   
        1,536    12b      2      256     12.6M     313.8 us   
        1,664    13b      2      256     12.6M     313.8 us   
        1,792    14b      2      256     12.6M     313.8 us   
        1,920    15b      2      256     12.6M     313.8 us   
        2,048    16b      2      256     12.6M     313.8 us   
        2,176    17b      2      256     12.6M     313.8 us   
        2,304    18b      2      256     12.6M     313.8 us   
        2,432    19b      2      256     12.6M     313.8 us   
        2,560    20b      2      256     12.6M     313.8 us   
        2,688    21b      2      256     12.6M     313.8 us   
        2,816    22b      2      256     12.6M     313.8 us   
        2,944    23b      2      256     12.6M     313.8 us   
        3,072    24b      2      256     12.6M     313.8 us   
        3,200    25b      2      256     12.6M     313.8 us   
        3,328    26b      2      256     12.6M     313.8 us   
        3,456    27b      2      256     12.6M     313.8 us   
        3,584    28b      2      256     12.6M     313.8 us   
        3,712    29b      2      256     12.6M     313.8 us   
        3,840    30b      2      256     12.6M     313.8 us   
        3,968    31b      2      256     12.6M     313.8 us   
        4,096    32b      2      256     12.6M     313.8 us   
       32,768 direct      1    32768      1.1G    (too big)   

  N = 49,152   (max splits @ LEAF=b: 2)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      3        3    281.0M     291.8 us   min-TIME
          256     2b      3        3    281.0M     291.8 us   
          384     3b      2      384     25.2M     434.9 us   min-WORK
          512     4b      2      384     25.2M     434.9 us   
          640     5b      2      384     25.2M     434.9 us   
          768     6b      2      384     25.2M     434.9 us   
          896     7b      2      384     25.2M     434.9 us   
        1,024     8b      2      384     25.2M     434.9 us   
        1,152     9b      2      384     25.2M     434.9 us   
        1,280    10b      2      384     25.2M     434.9 us   
        1,408    11b      2      384     25.2M     434.9 us   
        1,536    12b      2      384     25.2M     434.9 us   
        1,664    13b      2      384     25.2M     434.9 us   
        1,792    14b      2      384     25.2M     434.9 us   
        1,920    15b      2      384     25.2M     434.9 us   
        2,048    16b      2      384     25.2M     434.9 us   
        2,176    17b      2      384     25.2M     434.9 us   
        2,304    18b      2      384     25.2M     434.9 us   
        2,432    19b      2      384     25.2M     434.9 us   
        2,560    20b      2      384     25.2M     434.9 us   
        2,688    21b      2      384     25.2M     434.9 us   
        2,816    22b      2      384     25.2M     434.9 us   
        2,944    23b      2      384     25.2M     434.9 us   
        3,072    24b      2      384     25.2M     434.9 us   
        3,200    25b      2      384     25.2M     434.9 us   
        3,328    26b      2      384     25.2M     434.9 us   
        3,456    27b      2      384     25.2M     434.9 us   
        3,584    28b      2      384     25.2M     434.9 us   
        3,712    29b      2      384     25.2M     434.9 us   
        3,840    30b      2      384     25.2M     434.9 us   
        3,968    31b      2      384     25.2M     434.9 us   
        4,096    32b      2      384     25.2M     434.9 us   
       49,152 direct      1    49152      2.4G    (too big)   

  N = 65,536   (max splits @ LEAF=b: 2)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      3        4    285.2M     336.8 us   min-TIME
          256     2b      3        4    285.2M     336.8 us   
          384     3b      3        4    285.2M     336.8 us   
          512     4b      2      512     41.9M     577.9 us   min-WORK
          640     5b      2      512     41.9M     577.9 us   
          768     6b      2      512     41.9M     577.9 us   
          896     7b      2      512     41.9M     577.9 us   
        1,024     8b      2      512     41.9M     577.9 us   
        1,152     9b      2      512     41.9M     577.9 us   
        1,280    10b      2      512     41.9M     577.9 us   
        1,408    11b      2      512     41.9M     577.9 us   
        1,536    12b      2      512     41.9M     577.9 us   
        1,664    13b      2      512     41.9M     577.9 us   
        1,792    14b      2      512     41.9M     577.9 us   
        1,920    15b      2      512     41.9M     577.9 us   
        2,048    16b      2      512     41.9M     577.9 us   
        2,176    17b      2      512     41.9M     577.9 us   
        2,304    18b      2      512     41.9M     577.9 us   
        2,432    19b      2      512     41.9M     577.9 us   
        2,560    20b      2      512     41.9M     577.9 us   
        2,688    21b      2      512     41.9M     577.9 us   
        2,816    22b      2      512     41.9M     577.9 us   
        2,944    23b      2      512     41.9M     577.9 us   
        3,072    24b      2      512     41.9M     577.9 us   
        3,200    25b      2      512     41.9M     577.9 us   
        3,328    26b      2      512     41.9M     577.9 us   
        3,456    27b      2      512     41.9M     577.9 us   
        3,584    28b      2      512     41.9M     577.9 us   
        3,712    29b      2      512     41.9M     577.9 us   
        3,840    30b      2      512     41.9M     577.9 us   
        3,968    31b      2      512     41.9M     577.9 us   
        4,096    32b      2      512     41.9M     577.9 us   
       65,536 direct      1    65536      4.3G    (too big)   
 N = 98,304   (max splits @ LEAF=b: 2)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      3        6    293.6M     337.5 us   min-TIME
          256     2b      3        6    293.6M     337.5 us   
          384     3b      3        6    293.6M     337.5 us   
          512     4b      3        6    293.6M     337.5 us   
          640     5b      3        6    293.6M     337.5 us   
          768     6b      2      768     88.1M     970.6 us   min-WORK
          896     7b      2      768     88.1M     970.6 us   
        1,024     8b      2      768     88.1M     970.6 us   
        1,152     9b      2      768     88.1M     970.6 us   
        1,280    10b      2      768     88.1M     970.6 us   
        1,408    11b      2      768     88.1M     970.6 us   
        1,536    12b      2      768     88.1M     970.6 us   
        1,664    13b      2      768     88.1M     970.6 us   
        1,792    14b      2      768     88.1M     970.6 us   
        1,920    15b      2      768     88.1M     970.6 us   
        2,048    16b      2      768     88.1M     970.6 us   
        2,176    17b      2      768     88.1M     970.6 us   
        2,304    18b      2      768     88.1M     970.6 us   
        2,432    19b      2      768     88.1M     970.6 us   
        2,560    20b      2      768     88.1M     970.6 us   
        2,688    21b      2      768     88.1M     970.6 us   
        2,816    22b      2      768     88.1M     970.6 us   
        2,944    23b      2      768     88.1M     970.6 us   
        3,072    24b      2      768     88.1M     970.6 us   
        3,200    25b      2      768     88.1M     970.6 us   
        3,328    26b      2      768     88.1M     970.6 us   
        3,456    27b      2      768     88.1M     970.6 us   
        3,584    28b      2      768     88.1M     970.6 us   
        3,712    29b      2      768     88.1M     970.6 us   
        3,840    30b      2      768     88.1M     970.6 us   
        3,968    31b      2      768     88.1M     970.6 us   
        4,096    32b      2      768     88.1M     970.6 us   
       98,304 direct      1    98304      9.7G    (too big)   

  N = 131,072   (max splits @ LEAF=b: 2)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      3        8    302.0M     392.4 us   min-TIME
          256     2b      3        8    302.0M     392.4 us   
          384     3b      3        8    302.0M     392.4 us   
          512     4b      3        8    302.0M     392.4 us   
          640     5b      3        8    302.0M     392.4 us   
          768     6b      3        8    302.0M     392.4 us   
          896     7b      3        8    302.0M     392.4 us   
        1,024     8b      2     1024    151.0M      1.78 ms   min-WORK
        1,152     9b      2     1024    151.0M      1.78 ms   
        1,280    10b      2     1024    151.0M      1.78 ms   
        1,408    11b      2     1024    151.0M      1.78 ms   
        1,536    12b      2     1024    151.0M      1.78 ms   
        1,664    13b      2     1024    151.0M      1.78 ms   
        1,792    14b      2     1024    151.0M      1.78 ms   
        1,920    15b      2     1024    151.0M      1.78 ms   
        2,048    16b      2     1024    151.0M      1.78 ms   
        2,176    17b      2     1024    151.0M      1.78 ms   
        2,304    18b      2     1024    151.0M      1.78 ms   
        2,432    19b      2     1024    151.0M      1.78 ms   
        2,560    20b      2     1024    151.0M      1.78 ms   
        2,688    21b      2     1024    151.0M      1.78 ms   
        2,816    22b      2     1024    151.0M      1.78 ms   
        2,944    23b      2     1024    151.0M      1.78 ms   
        3,072    24b      2     1024    151.0M      1.78 ms   
        3,200    25b      2     1024    151.0M      1.78 ms   
        3,328    26b      2     1024    151.0M      1.78 ms   
        3,456    27b      2     1024    151.0M      1.78 ms   
        3,584    28b      2     1024    151.0M      1.78 ms   
        3,712    29b      2     1024    151.0M      1.78 ms   
        3,840    30b      2     1024    151.0M      1.78 ms   
        3,968    31b      2     1024    151.0M      1.78 ms   
        4,096    32b      2     1024    151.0M      1.78 ms   
      131,072 direct      1   131072     17.2G    (too big)   

  N = 196,608   (max splits @ LEAF=b: 2)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      3       12    318.8M     433.5 us   min-WORK min-TIME
          256     2b      3       12    318.8M     433.5 us   
          384     3b      3       12    318.8M     433.5 us   
          512     4b      3       12    318.8M     433.5 us   
          640     5b      3       12    318.8M     433.5 us   
          768     6b      3       12    318.8M     433.5 us   
          896     7b      3       12    318.8M     433.5 us   
        1,024     8b      3       12    318.8M     433.5 us   
        1,152     9b      3       12    318.8M     433.5 us   
        1,280    10b      3       12    318.8M     433.5 us   
        1,408    11b      3       12    318.8M     433.5 us   
        1,536    12b      2     1536    327.2M      4.35 ms   
        1,664    13b      2     1536    327.2M      4.35 ms   
        1,792    14b      2     1536    327.2M      4.35 ms   
        1,920    15b      2     1536    327.2M      4.35 ms   
        2,048    16b      2     1536    327.2M      4.35 ms   
        2,176    17b      2     1536    327.2M      4.35 ms   
        2,304    18b      2     1536    327.2M      4.35 ms   
        2,432    19b      2     1536    327.2M      4.35 ms   
        2,560    20b      2     1536    327.2M      4.35 ms   
        2,688    21b      2     1536    327.2M      4.35 ms   
        2,816    22b      2     1536    327.2M      4.35 ms   
        2,944    23b      2     1536    327.2M      4.35 ms   
        3,072    24b      2     1536    327.2M      4.35 ms   
        3,200    25b      2     1536    327.2M      4.35 ms   
        3,328    26b      2     1536    327.2M      4.35 ms   
        3,456    27b      2     1536    327.2M      4.35 ms   
        3,584    28b      2     1536    327.2M      4.35 ms   
        3,712    29b      2     1536    327.2M      4.35 ms   
        3,840    30b      2     1536    327.2M      4.35 ms   
        3,968    31b      2     1536    327.2M      4.35 ms   
        4,096    32b      2     1536    327.2M      4.35 ms   
      196,608 direct      1   196608     38.7G    (too big)   

  N = 262,144   (max splits @ LEAF=b: 2)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      3       16    335.5M     538.1 us   min-WORK min-TIME
          256     2b      3       16    335.5M     538.1 us   
          384     3b      3       16    335.5M     538.1 us   
          512     4b      3       16    335.5M     538.1 us   
          640     5b      3       16    335.5M     538.1 us   
          768     6b      3       16    335.5M     538.1 us   
          896     7b      3       16    335.5M     538.1 us   
        1,024     8b      3       16    335.5M     538.1 us   
        1,152     9b      3       16    335.5M     538.1 us   
        1,280    10b      3       16    335.5M     538.1 us   
        1,408    11b      3       16    335.5M     538.1 us   
        1,536    12b      3       16    335.5M     538.1 us   
        1,664    13b      3       16    335.5M     538.1 us   
        1,792    14b      3       16    335.5M     538.1 us   
        1,920    15b      3       16    335.5M     538.1 us   
        2,048    16b      2     2048    570.4M      8.88 ms   
        2,176    17b      2     2048    570.4M      8.88 ms   
        2,304    18b      2     2048    570.4M      8.88 ms   
        2,432    19b      2     2048    570.4M      8.88 ms   
        2,560    20b      2     2048    570.4M      8.88 ms   
        2,688    21b      2     2048    570.4M      8.88 ms   
        2,816    22b      2     2048    570.4M      8.88 ms   
        2,944    23b      2     2048    570.4M      8.88 ms   
        3,072    24b      2     2048    570.4M      8.88 ms   
        3,200    25b      2     2048    570.4M      8.88 ms   
        3,328    26b      2     2048    570.4M      8.88 ms   
        3,456    27b      2     2048    570.4M      8.88 ms   
        3,584    28b      2     2048    570.4M      8.88 ms   
        3,712    29b      2     2048    570.4M      8.88 ms   
        3,840    30b      2     2048    570.4M      8.88 ms   
        3,968    31b      2     2048    570.4M      8.88 ms   
        4,096    32b      2     2048    570.4M      8.88 ms   
      262,144 direct      1   262144     68.7G    (too big)   

  N = 393,216   (max splits @ LEAF=b: 2)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      3       24    369.1M     642.7 us   min-WORK min-TIME
          256     2b      3       24    369.1M     642.7 us   
          384     3b      3       24    369.1M     642.7 us   
          512     4b      3       24    369.1M     642.7 us   
          640     5b      3       24    369.1M     642.7 us   
          768     6b      3       24    369.1M     642.7 us   
          896     7b      3       24    369.1M     642.7 us   
        1,024     8b      3       24    369.1M     642.7 us   
        1,152     9b      3       24    369.1M     642.7 us   
        1,280    10b      3       24    369.1M     642.7 us   
        1,408    11b      3       24    369.1M     642.7 us   
        1,536    12b      3       24    369.1M     642.7 us   
        1,664    13b      3       24    369.1M     642.7 us   
        1,792    14b      3       24    369.1M     642.7 us   
        1,920    15b      3       24    369.1M     642.7 us   
        2,048    16b      3       24    369.1M     642.7 us   
        2,176    17b      3       24    369.1M     642.7 us   
        2,304    18b      3       24    369.1M     642.7 us   
        2,432    19b      3       24    369.1M     642.7 us   
        2,560    20b      3       24    369.1M     642.7 us   
        2,688    21b      3       24    369.1M     642.7 us   
        2,816    22b      3       24    369.1M     642.7 us   
        2,944    23b      3       24    369.1M     642.7 us   
        3,072    24b      2     3072      1.3G     18.87 ms   
        3,200    25b      2     3072      1.3G     18.87 ms   
        3,328    26b      2     3072      1.3G     18.87 ms   
        3,456    27b      2     3072      1.3G     18.87 ms   
        3,584    28b      2     3072      1.3G     18.87 ms   
        3,712    29b      2     3072      1.3G     18.87 ms   
        3,840    30b      2     3072      1.3G     18.87 ms   
        3,968    31b      2     3072      1.3G     18.87 ms   
        4,096    32b      2     3072      1.3G     18.87 ms   
      393,216 direct      1   393216    154.6G    (too big)   

  N = 524,288   (max splits @ LEAF=b: 2)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      3       32    402.7M     834.4 us   min-WORK min-TIME
          256     2b      3       32    402.7M     834.4 us   
          384     3b      3       32    402.7M     834.4 us   
          512     4b      3       32    402.7M     834.4 us   
          640     5b      3       32    402.7M     834.4 us   
          768     6b      3       32    402.7M     834.4 us   
          896     7b      3       32    402.7M     834.4 us   
        1,024     8b      3       32    402.7M     834.4 us   
        1,152     9b      3       32    402.7M     834.4 us   
        1,280    10b      3       32    402.7M     834.4 us   
        1,408    11b      3       32    402.7M     834.4 us   
        1,536    12b      3       32    402.7M     834.4 us   
        1,664    13b      3       32    402.7M     834.4 us   
        1,792    14b      3       32    402.7M     834.4 us   
        1,920    15b      3       32    402.7M     834.4 us   
        2,048    16b      3       32    402.7M     834.4 us   
        2,176    17b      3       32    402.7M     834.4 us   
        2,304    18b      3       32    402.7M     834.4 us   
        2,432    19b      3       32    402.7M     834.4 us   
        2,560    20b      3       32    402.7M     834.4 us   
        2,688    21b      3       32    402.7M     834.4 us   
        2,816    22b      3       32    402.7M     834.4 us   
        2,944    23b      3       32    402.7M     834.4 us   
        3,072    24b      3       32    402.7M     834.4 us   
        3,200    25b      3       32    402.7M     834.4 us   
        3,328    26b      3       32    402.7M     834.4 us   
        3,456    27b      3       32    402.7M     834.4 us   
        3,584    28b      3       32    402.7M     834.4 us   
        3,712    29b      3       32    402.7M     834.4 us   
        3,840    30b      3       32    402.7M     834.4 us   
        3,968    31b      3       32    402.7M     834.4 us   
        4,096    32b      2     4096      2.2G     34.68 ms   
      524,288 direct      1   524288    274.9G    (too big)   

################################################################################################
#  3 splits  (N = b^3 * k)
################################################################################################

  N = 2,097,152   (max splits @ LEAF=b: 2)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      3      128    805.3M      2.88 ms   min-WORK min-TIME
          256     2b      3      128    805.3M      2.88 ms   
          384     3b      3      128    805.3M      2.88 ms   
          512     4b      3      128    805.3M      2.88 ms   
          640     5b      3      128    805.3M      2.88 ms   
          768     6b      3      128    805.3M      2.88 ms   
          896     7b      3      128    805.3M      2.88 ms   
        1,024     8b      3      128    805.3M      2.88 ms   
        1,152     9b      3      128    805.3M      2.88 ms   
        1,280    10b      3      128    805.3M      2.88 ms   
        1,408    11b      3      128    805.3M      2.88 ms   
        1,536    12b      3      128    805.3M      2.88 ms   
        1,664    13b      3      128    805.3M      2.88 ms   
        1,792    14b      3      128    805.3M      2.88 ms   
        1,920    15b      3      128    805.3M      2.88 ms   
        2,048    16b      3      128    805.3M      2.88 ms   
        2,176    17b      3      128    805.3M      2.88 ms   
        2,304    18b      3      128    805.3M      2.88 ms   
        2,432    19b      3      128    805.3M      2.88 ms   
        2,560    20b      3      128    805.3M      2.88 ms   
        2,688    21b      3      128    805.3M      2.88 ms   
        2,816    22b      3      128    805.3M      2.88 ms   
        2,944    23b      3      128    805.3M      2.88 ms   
        3,072    24b      3      128    805.3M      2.88 ms   
        3,200    25b      3      128    805.3M      2.88 ms   
        3,328    26b      3      128    805.3M      2.88 ms   
        3,456    27b      3      128    805.3M      2.88 ms   
        3,584    28b      3      128    805.3M      2.88 ms   
        3,712    29b      3      128    805.3M      2.88 ms   
        3,840    30b      3      128    805.3M      2.88 ms   
        3,968    31b      3      128    805.3M      2.88 ms   
        4,096    32b      3      128    805.3M      2.88 ms   
    2,097,152 direct      1  2097152      4.4T    (too big)   

  N = 4,194,304   (max splits @ LEAF=b: 3)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      4        2     36.0G      4.10 ms   min-TIME
          256     2b      3      256      2.1G      4.10 ms   min-WORK
          384     3b      3      256      2.1G      4.10 ms   
          512     4b      3      256      2.1G      4.10 ms   
          640     5b      3      256      2.1G      4.10 ms   
          768     6b      3      256      2.1G      4.10 ms   
          896     7b      3      256      2.1G      4.10 ms   
        1,024     8b      3      256      2.1G      4.10 ms   
        1,152     9b      3      256      2.1G      4.10 ms   
        1,280    10b      3      256      2.1G      4.10 ms   
        1,408    11b      3      256      2.1G      4.10 ms   
        1,536    12b      3      256      2.1G      4.10 ms   
        1,664    13b      3      256      2.1G      4.10 ms   
        1,792    14b      3      256      2.1G      4.10 ms   
        1,920    15b      3      256      2.1G      4.10 ms   
        2,048    16b      3      256      2.1G      4.10 ms   
        2,176    17b      3      256      2.1G      4.10 ms   
        2,304    18b      3      256      2.1G      4.10 ms   
        2,432    19b      3      256      2.1G      4.10 ms   
        2,560    20b      3      256      2.1G      4.10 ms   
        2,688    21b      3      256      2.1G      4.10 ms   
        2,816    22b      3      256      2.1G      4.10 ms   
        2,944    23b      3      256      2.1G      4.10 ms   
        3,072    24b      3      256      2.1G      4.10 ms   
        3,200    25b      3      256      2.1G      4.10 ms   
        3,328    26b      3      256      2.1G      4.10 ms   
        3,456    27b      3      256      2.1G      4.10 ms   
        3,584    28b      3      256      2.1G      4.10 ms   
        3,712    29b      3      256      2.1G      4.10 ms   
        3,840    30b      3      256      2.1G      4.10 ms   
        3,968    31b      3      256      2.1G      4.10 ms   
        4,096    32b      3      256      2.1G      4.10 ms   
    4,194,304 direct      1  4194304     17.6T    (too big)   

  N = 8,388,608   (max splits @ LEAF=b: 3)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      4        4     37.6G      8.37 ms   min-TIME
          256     2b      4        4     37.6G      8.37 ms   
          384     3b      4        4     37.6G      8.37 ms   
          512     4b      3      512      6.4G      9.13 ms   min-WORK
          640     5b      3      512      6.4G      9.13 ms   
          768     6b      3      512      6.4G      9.13 ms   
          896     7b      3      512      6.4G      9.13 ms   
        1,024     8b      3      512      6.4G      9.13 ms   
        1,152     9b      3      512      6.4G      9.13 ms   
        1,280    10b      3      512      6.4G      9.13 ms   
        1,408    11b      3      512      6.4G      9.13 ms   
        1,536    12b      3      512      6.4G      9.13 ms   
        1,664    13b      3      512      6.4G      9.13 ms   
        1,792    14b      3      512      6.4G      9.13 ms   
        1,920    15b      3      512      6.4G      9.13 ms   
        2,048    16b      3      512      6.4G      9.13 ms   
        2,176    17b      3      512      6.4G      9.13 ms   
        2,304    18b      3      512      6.4G      9.13 ms   
        2,432    19b      3      512      6.4G      9.13 ms   
        2,560    20b      3      512      6.4G      9.13 ms   
        2,688    21b      3      512      6.4G      9.13 ms   
        2,816    22b      3      512      6.4G      9.13 ms   
        2,944    23b      3      512      6.4G      9.13 ms   
        3,072    24b      3      512      6.4G      9.13 ms   
        3,200    25b      3      512      6.4G      9.13 ms   
        3,328    26b      3      512      6.4G      9.13 ms   
        3,456    27b      3      512      6.4G      9.13 ms   
        3,584    28b      3      512      6.4G      9.13 ms   
        3,712    29b      3      512      6.4G      9.13 ms   
        3,840    30b      3      512      6.4G      9.13 ms   
        3,968    31b      3      512      6.4G      9.13 ms   
        4,096    32b      3      512      6.4G      9.13 ms   
    8,388,608 direct      1  8388608     70.4T    (too big)   

  N = 16,777,216   (max splits @ LEAF=b: 3)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      4        8     40.8G     19.36 ms   min-TIME
          256     2b      4        8     40.8G     19.36 ms   
          384     3b      4        8     40.8G     19.36 ms   
          512     4b      4        8     40.8G     19.36 ms   
          640     5b      4        8     40.8G     19.36 ms   
          768     6b      4        8     40.8G     19.36 ms   
          896     7b      4        8     40.8G     19.36 ms   
        1,024     8b      3     1024     21.5G     22.24 ms   min-WORK
        1,152     9b      3     1024     21.5G     22.24 ms   
        1,280    10b      3     1024     21.5G     22.24 ms   
        1,408    11b      3     1024     21.5G     22.24 ms   
        1,536    12b      3     1024     21.5G     22.24 ms   
        1,664    13b      3     1024     21.5G     22.24 ms   
        1,792    14b      3     1024     21.5G     22.24 ms   
        1,920    15b      3     1024     21.5G     22.24 ms   
        2,048    16b      3     1024     21.5G     22.24 ms   
        2,176    17b      3     1024     21.5G     22.24 ms   
        2,304    18b      3     1024     21.5G     22.24 ms   
        2,432    19b      3     1024     21.5G     22.24 ms   
        2,560    20b      3     1024     21.5G     22.24 ms   
        2,688    21b      3     1024     21.5G     22.24 ms   
        2,816    22b      3     1024     21.5G     22.24 ms   
        2,944    23b      3     1024     21.5G     22.24 ms   
        3,072    24b      3     1024     21.5G     22.24 ms   
        3,200    25b      3     1024     21.5G     22.24 ms   
        3,328    26b      3     1024     21.5G     22.24 ms   
        3,456    27b      3     1024     21.5G     22.24 ms   
        3,584    28b      3     1024     21.5G     22.24 ms   
        3,712    29b      3     1024     21.5G     22.24 ms   
        3,840    30b      3     1024     21.5G     22.24 ms   
        3,968    31b      3     1024     21.5G     22.24 ms   
        4,096    32b      3     1024     21.5G     22.24 ms   
    16,777,216 direct      1 16777216    281.5T    (too big)   

  N = 33,554,432   (max splits @ LEAF=b: 3)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      4       16     47.2G     41.94 ms   min-WORK min-TIME
          256     2b      4       16     47.2G     41.94 ms   
          384     3b      4       16     47.2G     41.94 ms   
          512     4b      4       16     47.2G     41.94 ms   
          640     5b      4       16     47.2G     41.94 ms   
          768     6b      4       16     47.2G     41.94 ms   
          896     7b      4       16     47.2G     41.94 ms   
        1,024     8b      4       16     47.2G     41.94 ms   
        1,152     9b      4       16     47.2G     41.94 ms   
        1,280    10b      4       16     47.2G     41.94 ms   
        1,408    11b      4       16     47.2G     41.94 ms   
        1,536    12b      4       16     47.2G     41.94 ms   
        1,664    13b      4       16     47.2G     41.94 ms   
        1,792    14b      4       16     47.2G     41.94 ms   
        1,920    15b      4       16     47.2G     41.94 ms   
        2,048    16b      3     2048     77.3G     54.75 ms   
        2,176    17b      3     2048     77.3G     54.75 ms   
        2,304    18b      3     2048     77.3G     54.75 ms   
        2,432    19b      3     2048     77.3G     54.75 ms   
        2,560    20b      3     2048     77.3G     54.75 ms   
        2,688    21b      3     2048     77.3G     54.75 ms   
        2,816    22b      3     2048     77.3G     54.75 ms   
        2,944    23b      3     2048     77.3G     54.75 ms   
        3,072    24b      3     2048     77.3G     54.75 ms   
        3,200    25b      3     2048     77.3G     54.75 ms   
        3,328    26b      3     2048     77.3G     54.75 ms   
        3,456    27b      3     2048     77.3G     54.75 ms   
        3,584    28b      3     2048     77.3G     54.75 ms   
        3,712    29b      3     2048     77.3G     54.75 ms   
        3,840    30b      3     2048     77.3G     54.75 ms   
        3,968    31b      3     2048     77.3G     54.75 ms   
        4,096    32b      3     2048     77.3G     54.75 ms   
    33,554,432 direct      1 33554432      1.1P    (too big)   

  N = 67,108,864   (max splits @ LEAF=b: 3)
         LEAF   =k*b levels  leaf_sz      work         time   best
          128     1b      4       32     60.1G     83.55 ms   min-WORK min-TIME
          256     2b      4       32     60.1G     83.55 ms   
          384     3b      4       32     60.1G     83.55 ms   
          512     4b      4       32     60.1G     83.55 ms   
          640     5b      4       32     60.1G     83.55 ms   
          768     6b      4       32     60.1G     83.55 ms   
          896     7b      4       32     60.1G     83.55 ms   
        1,024     8b      4       32     60.1G     83.55 ms   
        1,152     9b      4       32     60.1G     83.55 ms   
        1,280    10b      4       32     60.1G     83.55 ms   
        1,408    11b      4       32     60.1G     83.55 ms   
        1,536    12b      4       32     60.1G     83.55 ms   
        1,664    13b      4       32     60.1G     83.55 ms   
        1,792    14b      4       32     60.1G     83.55 ms   
        1,920    15b      4       32     60.1G     83.55 ms   
        2,048    16b      4       32     60.1G     83.55 ms   
        2,176    17b      4       32     60.1G     83.55 ms   
        2,304    18b      4       32     60.1G     83.55 ms   
        2,432    19b      4       32     60.1G     83.55 ms   
        2,560    20b      4       32     60.1G     83.55 ms   
        2,688    21b      4       32     60.1G     83.55 ms   
        2,816    22b      4       32     60.1G     83.55 ms   
        2,944    23b      4       32     60.1G     83.55 ms   
        3,072    24b      4       32     60.1G     83.55 ms   
        3,200    25b      4       32     60.1G     83.55 ms   
        3,328    26b      4       32     60.1G     83.55 ms   
        3,456    27b      4       32     60.1G     83.55 ms   
        3,584    28b      4       32     60.1G     83.55 ms   
        3,712    29b      4       32     60.1G     83.55 ms   
        3,840    30b      4       32     60.1G     83.55 ms   
        3,968    31b      4       32     60.1G     83.55 ms   
        4,096    32b      3     4096    292.1G    121.47 ms   
    67,108,864 direct      1 67108864      4.5P    (too big) 