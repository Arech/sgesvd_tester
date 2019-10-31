# Isolated code to make issue with OpenBLAS::LAPACKE_sgesvd() appear

On how to compile the proper OpenBLAS binary see [the issue](https://github.com/xianyi/OpenBLAS/issues/2297).


VC2015 is used to build the project, though probably any modern compiler will work, just update building instructions to make it build this simple x64 console app.

1. In the `Project Property pages` update section `VC++ Directories`:

     1. set correct path to OpenBLAS distro's `./include` folder in `Include Directories`
     2. set correct path to OpenBLAS distro's `./lib` folder in `Library Directories`
2. compile the project to get `.exe`

3. copy the properly compiled `libopenblas.dll` in question and all the necessary mingw-w64 env .dll's (at least you would need `libgfortran-#.dll` and its dependecies) near the project's `.exe`

4. run `.exe`

What to expect:

- with a normal OpenBLAS binary (either [supplied](https://sourceforge.net/projects/openblas/files/) by developers or compiled with mingw-w64 v.6.3) all four tests should pass (double and float for default fp state and with rounding).
- with OpenBLAS compiled using mingw-w64 v8.3 test with double will pass, but test with float will fail due to `sgesvd()` returning an error (in default fp state mode) or returning success with a junk in output variables (when fp state changed to round towards zero). The last option is exactly what I [observed](https://github.com/xianyi/OpenBLAS/issues/2297).
