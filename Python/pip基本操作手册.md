# 1. 换源

在使用 pip 安装 Python 包时，有时会遇到下载速度慢或无法访问的问题。为了解决这些问题，可以更换 pip 的下载源到国内的镜像源。以下是更换 pip 源的常用方法：
临时更换源
在安装包时，可以使用 -i 选项指定临时的镜像源：
pip install {package_name} -i https://pypi.tuna.tsinghua.edu.cn/simple/

常用国内镜像源列表
1. 阿里云
  - URL: https://mirrors.aliyun.com/pypi/simple/
2. 清华大学
  - URL: https://pypi.tuna.tsinghua.edu.cn/simple/
3. 中国科学技术大学 (USTC)
  - URL: https://pypi.mirrors.ustc.edu.cn/simple/
4. 豆瓣 (Douban)
  - URL: https://pypi.douban.com/simple/

# 永久更换源
可以通过修改 pip 的配置文件来永久更换源。以下是具体步骤：
## Linux 或 macOS
1. 打开或创建 pip 配置文件：
```bash
mkdir -p ~/.pip
vim ~/.pip/pip.conf
```
2. 在文件中添加以下内容，以使用清华大学的镜像源为例：
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple

## Windows
1. 打开或创建 pip 配置文件：
```bash
mkdir %HOMEPATH%\pip
notepad %HOMEPATH%\pip\pip.ini
```
2. 在文件中添加以下内容，以使用清华大学的镜像源为例：
```text
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

验证更换源是否成功
运行以下命令，检查 pip 是否使用了新的镜像源：
pip config list
这会显示当前 pip 的配置，确保 index-url 指向了你指定的镜像源。
通过以上步骤，你可以有效地更换 pip 的源，以获得更快的下载速度和更稳定的连接。