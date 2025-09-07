Mueen's Algorithm for Similarity Search (MASS)
==============================================

This module implements version 2 of Mueen's Algorithm for Similarity Search (MASS_V2). MASS is an algorithm to create Distance Profile of a query to a long time series: given a query subsequence `Q` of length _m_ and a long time-series `T` of length _n_, MASS computes the z-normalized Euclidean distance between `Q` and every subsequence in `T` of length _m_. 

The key idea underlying MASS is that z-normalized Euclidean distance reduces to a formula involving sliding dot products of `Q` against subsequences of `T` plus mean and variance terms. Sliding dot products can be computed in **O(n log n)** using fast convolution via FFT (Fast Fourier Transform). Consequently, MASS_V2 computes the entire Distance Profile in O(n log n), independent of data distribution.

Additionally, this module provides convenience functions to find the best match, or the top K matches, in the time-series. 


Citation
--------

Abdullah Mueen, Sheng Zhong, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy Viswanathan, Chetan Kumar Gupta and Eamonn Keogh (2022), "The Fastest Similarity Search Algorithm for Time Series Subsequences under Euclidean Distance." 

URL: http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html


Algorithm Complexity
--------------------

- **Time:** O(n log n) 
  - Dominated by FFT operations
- **Space:** O(n)
  - Requires memory for the entire time-series, plus FFT working space
