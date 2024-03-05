def test_import():
    from alibi_detect.utils.pydantic_v1 import BaseModel
    from alibi_detect.utils.pydantic_v1.fields import ModelField

    # Try and instantiate something from the import
    class X(BaseModel):
        y: int

    ModelField(
        name="y",
        type_=str,
        class_validators={},
        model_config=X.__config__,
    )
