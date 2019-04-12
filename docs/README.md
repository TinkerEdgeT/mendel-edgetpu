This directory holds the source files required to build the Edge TPU reference with Sphinx.

You can build these docs locally with the `docs` make target. Of course, this requires that
you install Sphinx and other Python dependencies:

    # We require Python3, so if that's not your default, first start a virtual environment:
    python3 -m venv ~/.my_venvs/coraldocs
    source ~/.my_venvs/coraldocs/bin/activate

    # Move to the python-tflite-source/edgetpu/ directory...

    # Install the doc build dependencies:
    pip install -r ../docs/requirements.txt

    # Make sure you have a library build, because the SWIG API build is required:
    sh build_library.sh

    # Build the docs:
    make docs

The results are output in `python-tflite-source/docs/_build/html/`.

For more information about the syntax in these RST files, see the [reStructuredText documentation](
http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).