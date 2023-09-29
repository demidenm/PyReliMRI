PyReliMRI Installation
-----------------------
You can install the PyReliMRI package directly from your terminal using *pip install* for a (A) specific tagged release \
or (B) latest release that is on PyPI

.. code-block:: bash

   # [A] specific tagged release
   pip install git+https://github.com/demidenm/PyReliMRI.git@v2.0.0

   # [B] latest release on PyPI
   pip install pyrelimri


If the installation is successful, you will see something along the lines of *Successfully installed PyReliMRI-2.0.0* into your terminal.

Once the package is installed, you can import the `pyrelimri` module using:

.. code-block:: python

   import pyrelimri


Alternatively, you can load a specific function from the available modules. For example, if you're only interested in \
calculating the similarity between 3D Nifti images, you can load

.. code-block:: python

   from pyrelimri import similarity

Once the module is loaded, the functions within the module can be used. You can check with suffix '?' what input \
is required to run the function.

.. code-block:: python

    # how to use
    similarity.image_similarity(imgfile1 = path_to_img, imgfile2 = path_to_img,
                                mask=  path_to_mash, thresh = 1.25, similarity_type = 'dice')
    # required input
    similarity.image_similarity?


Required dependencies
`````````````````````

While a number of calculations are performed manually, PyReliMRI depends on several packages that must be installed. \
The effort here isn't to reinvent the wheel but integrate as many tools. While the versions below are not required \
they were sufficient to run each script during testing.

-  Python>=>3.6
-  numpy>=1.2
-  scipy=>1.9
-  pandas=>1.4
-  nilearn=>0.9
-  nibabel=>4.0.2
-  sklearn=>1.0.2
