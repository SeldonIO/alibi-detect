def test_import():
    from alibi_detect.utils.pydantic_v1 import BaseModel

    # Try and instantiate something from the import
    class X(BaseModel):
        y: int
