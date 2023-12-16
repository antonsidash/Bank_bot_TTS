@echo on


rem checking the installation TensorFlow
python -c "import tensorflow" 2>nul
if %errorlevel% neq 0 (
    echo installing TensorFlow...
    pip install tensorflow
)

rem checking the installation PyYAML
python -c "import yaml" 2>nul
if %errorlevel% neq 0 (
    echo installing PyYAML...
    pip install PyYAML
)

rem checking the installation aiogram
python -c "import aiogram" 2>nul
if %errorlevel% neq 0 (
    echo installing aiogram...
    pip install aiogram
)

rem checking the installation emoji
python -c "import emoji" 2>nul
if %errorlevel% neq 0 (
    echo installing emoji...
    pip install emoji
)

rem checking the installation numpy
python -c "import numpy" 2>nul
if %errorlevel% neq 0 (
    echo installing numpy...
    pip install numpy
)


echo all packages installed
