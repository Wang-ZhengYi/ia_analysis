"""Notebook-derived workflow namespace.

Purpose
-------
This package preserves code exported from historical notebooks so analyses can
be reviewed, compared, and gradually refactored into stable modules.

Provides
--------
- A stable import location for generated notebook exports.
- A bridge between exploratory notebooks and reusable package functions.

Notes
-----
New production code should prefer the domain packages under ``ia_analysis`` and
use notebook exports mainly as references.
"""


