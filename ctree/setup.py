from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

# Define extensions
extensions = [
    Extension(
        "search_tree",  # Name of the module
        ["/zfsauton2/home/jiayuc2/Proj_3/ContBAMCP_SL/ctree/search_tree.pyx", "/zfsauton2/home/jiayuc2/Proj_3/ContBAMCP_SL/ctree/lib/cnode.cpp",
         "/zfsauton2/home/jiayuc2/Proj_3/ContBAMCP_SL/ctree/lib/cminimax.cpp"],  # Source files
        include_dirs=["/zfsauton2/home/jiayuc2/Proj_3/ContBAMCP_SL/ctree/lib/"],  # Include directories for header files
        language="c++",  # Specify the language
        extra_compile_args=["-std=c++11"],  # Optional: compile args, like C++ version
    )
]

# Setup function
setup(
    name="search_tree",
    ext_modules=cythonize(extensions)
)