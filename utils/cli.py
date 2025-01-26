from functools import wraps
from argparse import ArgumentParser
from typing import Type, TypeVar, Callable
import warnings

from pydantic import BaseModel, Field

T = TypeVar('T')

def clap(cls: Type[T]) -> Callable[..., T]:
    """Decorator that converts a class's type annotations into CLI arguments. Name is used to represent
       Command Line Argument Parser (CLAP) like the Rust library."""
    assert issubclass(cls, BaseModel), f"Class {cls.__name__} is not a Pydantic model. It is recommended to use Pydantic models for clap decorators."

    @wraps(cls)
    def wrapper(*args, **kwargs):
        parser = ArgumentParser(description=cls.__doc__)
        # Get class annotations and default values
        annotations = cls.__annotations__
        # Get descriptions from Pydantic fields if available
        descriptions = {}
        if hasattr(cls, 'model_fields'):
            descriptions = {
            name: field.description
            for name, field in cls.model_fields.items()
            if field.description is not None
            }

            defaults = {
                name: field.default
                for name, field in cls.model_fields.items()
                if field.description is not None
            }

        # Add arguments
        for name, type_hint in annotations.items():
            parser.add_argument(
                f'--{name}',
                type=type_hint,
                default=defaults.get(name),
                help=descriptions.get(name),
                required=name not in defaults
            )
        
        # Parse and create instance
        args = parser.parse_args()
        return cls(**args.__dict__)
    
    return wrapper


if __name__ == '__main__':
    @clap
    class MyArgs(BaseModel):
        a: str = Field(..., description="This is a required argument")
        b: str = Field(default='default_val',description="This is a required argument")
    
    args = MyArgs()
    print(f"a: {args.a}, b: {args.b}")
