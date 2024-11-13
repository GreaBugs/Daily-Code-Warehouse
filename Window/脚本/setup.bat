@echo off
setlocal

:: 使用脚本所在的目录作为基准路径
set SCRIPT_DIR=%~dp0

:: 设置安装包的相对路径
set ANACONDA_INSTALLER=%SCRIPT_DIR%\installer\Anaconda3-2023.07-0-Windows-x86_64.exe
set PYCHARM_INSTALLER=%SCRIPT_DIR%\installer\pycharm-professional-2024.2.4.exe

:: 设置安装路径
set ANACONDA_INSTALL_PATH=D:\Anaconda3
set PYCHARM_INSTALL_PATH=D:\PyCharm

:: 安装 Anaconda
if exist %ANACONDA_INSTALLER% (
    echo Installing Anaconda...
    %ANACONDA_INSTALLER% /S /AddToPath=1 /D=%ANACONDA_INSTALL_PATH%
    echo Anaconda installation complete.
) else (
    echo Anaconda installer not found.
    exit /b
)

:: 添加 Anaconda 到系统路径（需要重新启动生效）
setx PATH "%ANACONDA_INSTALL_PATH%;%ANACONDA_INSTALL_PATH%\Scripts;%PATH%"
set "conda_exe=%ANACONDA_INSTALL_PATH%\Scripts\conda.exe"
set CONDA_PATH=%ANACONDA_INSTALL_PATH%\Scripts\conda.exe"

:: 检查是否安装成功
if exist "%conda_exe%" (
    echo Anaconda installation verified.
) else (
    echo Anaconda installation failed.
    exit /b
)

:: 创建名为 "train" 的虚拟环境
echo Creating virtual environment "train"...
%conda_exe% create -y -n train python=3.8
echo Virtual environment "train" created successfully.

:: 安装 PyCharm
if exist %PYCHARM_INSTALLER% (
    echo Installing PyCharm...
    %PYCHARM_INSTALLER% /S /D=%PYCHARM_INSTALL_PATH%
    echo PyCharm installation complete.
) else (
    echo PyCharm installer not found.
)

echo Installation and setup complete.

setx PATH "%PATH%;D:\Anaconda3"
setx PATH "%PATH%;D:\Anaconda3\Scripts"
setx PATH "%PATH%;D:\Anaconda3\Library\bin"

setx PIPIP_INDEX_URL https://pypi.tuna.tsinghua.edu.cn/simple
echo Pip source has been changed to Tsinghua University.

conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set channel_priority strict
echo Conda source has been changed to Tsinghua University.
pause

endlocal


