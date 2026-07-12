def test_import_satx():
    import satx

def test_import_subpackages():
    import satx.data
    import satx.models
    import satx.engine
    import satx.eval
    import satx.analysis
    import satx.utils

def test_import_evaluation_api():
    from satx.eval import (
        generate_evaluation_artifact,
        plot_evaluation_results,
        print_evaluation_results,
    )

    assert callable(generate_evaluation_artifact)
    assert callable(plot_evaluation_results)
    assert callable(print_evaluation_results)