def test_lenstronomy_version():
    """
    tests the import of lenstronomy
    """
    import lenstronomy

    assert lenstronomy.__version__ == "1.10.4"
