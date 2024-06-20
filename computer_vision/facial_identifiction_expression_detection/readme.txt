There was no wxPython available for Mac so I had to download the source, build from source it and install

Get source package wxPython-4.2.1.tar.gz
from
https://pypi.org/project/wxPython/#files

1. extract it to wheels\wxPython-4.2.1
2. cd wheels\wxPython-4.2.1
3. To get help on the commands to be used to build and install this package
    python build.py
4. python build.py dox etg --nodoc sip build
5. To build wxPython package
   python build.py build_py
6. To install that package
   python build.py install_py




