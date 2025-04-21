::
:: File: cupcake.bat
:: Posizionabile in una cartella nel PATH su Windows
:: Permette di eseguire "cupcake" da terminale

@echo off
python "%~dp0\cupcake.py" %*
