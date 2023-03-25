PyReliMRI Installation
-----------------------
You can install the PyReliMRI package directly from your terminal using *pip install*

.. code-block:: bash

   pip install git+https://github.com/demidenm/PyReliMRI.git

If the installation is successful, you will see something along the lines of *Successfully installed PyReliMRI-1.0.0* in your terminal.

Once the package is installed, you can import the `pyrelimri` module using:

.. code-block:: python

   import pyrelimri


Alternatively, you can load specific function from the available scripts. For example, if I am only interested in calculating the image similarity, I can load

.. code-block:: python

   from pyrelimri import similarity

Once the script is loaded, I can use the functions within the script to use it or what information is required for my project.

.. code-block:: python

    # how to use
    similarity.image_similarity(imgfile1 = path_to_img, imgfile2 = path_to_img, mask=  path_to_mash, thresh = 1.25, similarity_type = 'dice')
    # required input
    similarity.image_similarity?

Required dependencies
`````````````````````

While a number of calculations are performed manually, PyReliMRI uses several packages that must be installed. While the versions below are not \
required they were sufficient to run each script during testing.

-  Python>=>3.6
-  numpy>=1.2
-  scipy=>1.9
-  pandas=>1.4
-  nilearn=>0.9
