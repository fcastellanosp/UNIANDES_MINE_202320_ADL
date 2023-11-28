@echo off
cd..
cd..
setlocal
set PROJECTPATH=%cd%
set PYTHONPATH=%PYTHON310_HOME%;%cd%
echo "Starting the app at '%PROJECTPATH%\src\front\main.py"
echo "Enviroment variable '%PYTHONPATH%"
%PYTHON310_HOME%\python -m streamlit run %PROJECTPATH%\main.py
endlocal