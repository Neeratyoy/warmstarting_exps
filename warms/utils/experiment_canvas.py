from pathlib import Path
import toml


class ExpCanvas:
    """A class to manage the configuration of an experiment.
    
    Simple interface that loads the configuration from a TOML file and splits it into a
    global configuration and an access point specific configuration. The access point is a 
    subset of the configuration that is used to run the experiment. The global configuration 
    acts as the default configuration that is used when the access point does not have the key.

    The `__getattr__` method is used to dynamically create properties for all keys in the config.
    One can override this property for a specific key by defining a method with the same name and 
    adding a `@property` decorator.
    
    For example: 
    `def train_template_path()` is defined with an explicit check specific to itself.

    Example .toml file:
    ```toml
    [global]
    data_root = "data"
    results_root = "results"
    train_template = "train_template.yaml"

    [user1]
    data_root = "data/user1"
    ```
    NOTE: To enforce conversion of directories to Path() objects, the key should contain one of 
        {path, root, dir} in the name.
    """

    DEFAULT = "global"

    def __init__(self, filepath: Path, access_point: str = None):
        self.filepath = filepath
        self.access_point = access_point if access_point else self.DEFAULT

        _config = toml.load(filepath)
        self.global_config = _config[self.DEFAULT]
        self.config = _config[self.access_point]

    @property
    def train_template(self):
        """Returns the path to the training template.

        Defaults to the global template path only if the access point does not have a the
        `train_template` key.
        Raises an error if the key exists but the file does not.
        """
        _key = "train_template"
        _template = self.config.get(_key, None)
        if _template is not None and not Path(_template).is_file():         
            raise ValueError(
                f"{_template} does not exist. Please double-check the configuration to run."
            )
        else:
            _template = self.global_config[_key]
        value = Path(self.config.get(_key, self.global_config[_key])).absolute()

        return value

    def __getattr__(self, name) -> tuple[int | float | str | Path]:
        """Dynamically creates properties for all keys in the config.
        """
        try:
            value = self.config.get(name, self.global_config.get(name))
            # NOTE: absolute values ensure that overloaded folder names and string name don't clash
            if self._check_if_path(name, value):
                value = Path(value)
            return value
        except KeyError:
            raise AttributeError(f"Attribute '{name}' not found in config.")
        
    def _check_if_path(self, key: str, value: str) -> bool:
        if value is None:
            return False
        if Path(value).exists():
            # if the absolute path exists, it is a path
            # can be False if a directory such as the output has not yet been created
            return True
        # if it doesn't exist yet,
        #  it is either a string, or
        #  it is a path that doesn't exist, such as a new output directory
        if "path" in key or "root" in key or "dir" in key:
            # if the key has a path-like name, it is a path
            return True
        return False
# end of file