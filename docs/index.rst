.. Ravnest documentation master file, created by
   sphinx-quickstart on Tue Apr  2 12:15:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|logo| **Welcome to Ravnest's documentation!**
##############################################

.. |logo| image:: _static/raven-black.png
   :scale: 50


Ravnest is an asynchronous decentralized training framework designed for large modern deep learning models. It leverages the compute power of regular heterogeneous PCs with limited resources connected across the internet to achieve favorable performance metrics. By clustering compute nodes with similar data transfer speeds and computing capabilities, Ravnest removes the requirement for each node to store the entire model, facilitating decentralized training.


.. |research_paper_link| raw:: html

   <a href="https://arxiv.org/abs/2401.01728" target="_blank" style="font-weight:bold">Research Paper</a>

|research_paper_link|

.. |github_link| raw:: html

   <a href="https://github.com/ravenprotocol/ravnest" target="_blank" style="font-weight:bold">GitHub</a>

|github_link|

.. warning::

   Ravnest is in the throes of active development. Brace yourselves for updates! What you see today may not be what you encounter tomorrow. 🌪️

.. toctree::
   :maxdepth: 3
   :caption: Table of Contents

   quickstart
   api
   roles
   train
   walkthrough
   contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/ravenprotocol/ravnest