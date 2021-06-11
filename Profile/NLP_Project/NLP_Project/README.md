## BUILD
1.  CPP
```
cd cpp
mkdir build
cd build
cmake ..
make
make install
```

2. Make test in CPP
```
cd cpp/tests
mkdir build
cd build
cmake ..
make
```
3. Make the pythob lib

replace the *USERNAME*
```
cd ..
cd build
cmake .. -DPYTHON_LIBRARY_DIR="/Users/USERNAME/opt/anaconda3/lib/python3.7/site-packages" -DPYTHON_EXECUTABLE="/Users/USERNAME/opt/anaconda3/bin/python3"
make
make install
```
## Requirement
1. pytorch 1.6
2. numpy
3. tqdm
4. matplotlib.pyplot
5. nltk
6. re

## Files
1. "Project/cpp/src" contains implementation of logistic regression and svm in CPP, "Project/python" contains the corresponding pybin11 code
2. "Prject/tests" contains examples of basic classifier on bert representation
3. "Project/scripts/NER_Model" contains all neural network and data processing

## Code Online
https://colab.research.google.com/drive/1r9WPw79UcC8OnBb8LedBGFWE4U_bcL8-?usp=sharing