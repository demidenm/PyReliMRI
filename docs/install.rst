PyReliMRI Installation
-----------------------
You can install the PyReliMRI package directly from your terminal using *pip install*

.. code-block:: bash

   pip install git+https://github.com/demidenm/PyReliMRI.git

If the installation is successful, you will see something along the lines of *Successfully installed PyReliMRI-1.0.0* in your terminal.

Once the package is installed, you can import the entire `imgreliability` module using:

.. code-block:: python

   import imgreliability


Alternatively, you can load specific function from a script. For example, if I am only interested in calculating the image similarity, you can load

.. code-block:: python

   from imgreliability import similarity

Once the script is loaded, you can use the functions within the script to use it or what information is required.

.. code-block:: python

    # how to use
    similarity.image_similarity(imgfile1 = path_to_img, imgfile2 = path_to_img, mask=  path_to_mash, thresh = 1.25, similarity_type = 'dice')
    # required input
    similarity.image_similarity?