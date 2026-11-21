from dinkster_assets.model import AssetRef
from dinkster_schema.body import ASSET_REF_CLASS


def test_lookup_asset_parameter_names_the_class_execute_receives() -> None:
    # The schema matches the annotation by module and name so it needs no
    # dependency on dinkster-assets; this keeps the string on the real class.
    assert (AssetRef.__module__, AssetRef.__qualname__) == ASSET_REF_CLASS
