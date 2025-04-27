def test_lenstronomy_version():
    """Tests the import of lenstronomy."""
    import lenstronomy

    assert lenstronomy.__version__ == "1.12.6"
