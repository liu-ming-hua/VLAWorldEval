Build a reproducible evaluation
========================================

VLAWorldEval connects robot policies and action-conditioned world models through
explicit protocols and isolated runtimes. Start with the CPU demo, choose a
compatible pair, and inspect every generated rollout.

.. note::

   Our vision is to make reproducible, scalable VLA closed-loop evaluation a
   practical complement to real-world testing as world models become more faithful
   to actions. Plausible video alone does not establish physical validity or task success.

Start here
----------

* :doc:`en/quickstart` — your first rollout, with no weights or GPUs.
* :doc:`en/installation` — isolated environments, model assets and mirrors.
* :doc:`en/compatibility` — choose from the 16 declared execution routes.
* :doc:`en/models` — inputs, outputs, action meanings and training domains.

Two evaluation routes
---------------------

**Policy closed loop.** A VLA predicts actions; a compatible world model generates
the next observation. Generated observations feed the next policy call.

**Recorded-action replay.** Run a world model without a VLA and compare its
predictions against recorded future frames, kept outside the inference inputs.

Both routes share planning, worker isolation, artifact writing and diagnostic tools.
Read :doc:`en/workflows` to execute them, and :doc:`en/reliability` before interpreting scores.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: English manual

   en/README
   en/quickstart
   en/installation
   en/compatibility
   en/models
   en/workflows
   en/artifacts
   en/architecture
   en/extending
   en/reliability
   en/troubleshooting
   en/release

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: 中文手册

   zh/README
   zh/quickstart
   zh/installation
   zh/compatibility
   zh/models
   zh/workflows
   zh/artifacts
   zh/architecture
   zh/extending
   zh/reliability
   zh/troubleshooting
   zh/release
