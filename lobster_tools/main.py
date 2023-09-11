from omegaconf import OmegaConf
import hydra
from demo_config import MainConfig, register_configs


register_configs()
@hydra.main(version_base=None, config_name="config")
def example_main(cfg: MainConfig) -> None:
    """I would have more specific name here. For example this file might be called create_arctic_dc.py and the main()
    function would be called create_arctic_db() or sth"""
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    example_main()