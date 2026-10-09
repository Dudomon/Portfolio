"""Thin Spark / Delta / Feature Engineering adapters.

This is the only package that needs a Databricks runtime. Each function moves
data between Delta tables and the pure-pandas logic elsewhere in the package,
mostly through ``mapInPandas`` so the distributed path runs the same code the
unit tests exercise.
"""
