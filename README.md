- in sintesi, col merge like avremmo complessità minore,, però la branch divergence e la non coalescence rendono meno efficiente il lavoro dei thread nello stesso warp, quindi ci mette di più
<br>




**Compilation**

**Windows**

- open the general makefile
- set the right nvcc_win and VS_PATH
- open visual studio console

- CHOOSE between 1) and 2):
1) set make temporary to the session -> set PATH=C:\tools\ucrt-make\bin;%PATH% <br>
2) set make permanently in enviroment variables -> system variables -> Path -> copy this **C:\msys64\usr\bin** 

make --version
nvcc --version
cd path\to\ParallelCountTriangles
make OS=windows 

**linux**
cd path/to/ParallelCountTriangles
make OS=linux 


**remove .lib .exp**

**windows**
del /s /q *.lib *.exp

**linux**
find . -type f -name "*.lib" -delete
find . -type f -name "*.exp" -delete