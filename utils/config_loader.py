import yaml


def load_config_yaml(config_file_path: str = "config/config.yaml"):
    """This function loads the config file

    Args:
        config_file_path (str, optional): _description_. Defaults to "config\config.yaml".

    Returns:
        _type_: _description_
    """
    with open(config_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    # print(config)
    return config


# for testing load config, uncomment and run this file on terminal
# load_config_yaml("config\config.yaml")
